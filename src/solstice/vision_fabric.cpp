#include "rlf/solstice/vision_fabric.hpp"

#include <algorithm>
#include <array>
#include <bit>
#include <cmath>
#include <cctype>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <fstream>
#include <iomanip>
#include <limits>
#include <memory>
#include <queue>
#include <setjmp.h>
#include <sstream>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <utility>

#if defined(RLF_HAS_LIBPNG)
#include <png.h>
#endif

#if defined(RLF_HAS_LIBJPEG)
extern "C" {
#include <jpeglib.h>
}
#endif

#if defined(_WIN32)
#ifndef NOMINMAX
#define NOMINMAX
#endif
#include <windows.h>
#include <wincodec.h>
#endif

namespace rlf::solstice {
namespace {

constexpr std::uint64_t fnv_offset_basis = 14'695'981'039'346'656'037ULL;
constexpr std::uint64_t fnv_prime = 1'099'511'628'211ULL;
constexpr std::size_t legacy_descriptor_size = 16U;
constexpr std::size_t frontier_descriptor_size = 32U;

[[nodiscard]] bool incremental_sparse_router_updates_from_environment() {
    const char* const value = std::getenv("RLF_SPARSE_ROUTER_UPDATE_POLICY");
    if (value == nullptr || std::string_view(value).empty() ||
        std::string_view(value) == "incremental") {
        return true;
    }
    if (std::string_view(value) == "rebuild") {
        return false;
    }
    throw std::invalid_argument(
        "RLF_SPARSE_ROUTER_UPDATE_POLICY must be incremental or rebuild"
    );
}

[[nodiscard]] bool batched_sparse_reranking_from_environment() {
    const char* const value = std::getenv("RLF_SPARSE_RERANK_POLICY");
    if (value == nullptr || std::string_view(value).empty()) {
        return false;
    }
    if (std::string_view(value) == "batched") {
        return true;
    }
    if (std::string_view(value) == "per_query") {
        return false;
    }
    throw std::invalid_argument(
        "RLF_SPARSE_RERANK_POLICY must be batched or per_query"
    );
}

[[nodiscard]] bool sparse_rerank_policy_explicit_from_environment() noexcept {
    const char* const value = std::getenv("RLF_SPARSE_RERANK_POLICY");
    return value != nullptr && !std::string_view(value).empty();
}

[[nodiscard]] bool indexed_concept_updates_from_environment() {
    const char* const value = std::getenv("RLF_CONCEPT_UPDATE_POLICY");
    if (value == nullptr || std::string_view(value).empty() ||
        std::string_view(value) == "indexed") {
        return true;
    }
    if (std::string_view(value) == "linear") {
        return false;
    }
    throw std::invalid_argument(
        "RLF_CONCEPT_UPDATE_POLICY must be indexed or linear"
    );
}

[[nodiscard]] bool indexed_example_duplicate_lookup_from_environment() {
    const char* const value = std::getenv("RLF_EXAMPLE_DUPLICATE_POLICY");
    if (value == nullptr || std::string_view(value).empty() ||
        std::string_view(value) == "indexed") {
        return true;
    }
    if (std::string_view(value) == "linear") {
        return false;
    }
    throw std::invalid_argument(
        "RLF_EXAMPLE_DUPLICATE_POLICY must be indexed or linear"
    );
}

[[nodiscard]] bool persistent_mode_id_index_from_environment() {
    const char* const value = std::getenv("RLF_MODE_ID_INDEX_POLICY");
    if (value == nullptr || std::string_view(value).empty() ||
        std::string_view(value) == "persistent") {
        return true;
    }
    if (std::string_view(value) == "rebuild") {
        return false;
    }
    throw std::invalid_argument(
        "RLF_MODE_ID_INDEX_POLICY must be persistent or rebuild"
    );
}

void hash_u64(std::uint64_t& hash, const std::uint64_t value) noexcept {
    for (unsigned int index = 0U; index < 8U; ++index) {
        hash ^= (value >> (index * 8U)) & 0xFFULL;
        hash *= fnv_prime;
    }
}

void hash_string(std::uint64_t& hash, const std::string_view value) noexcept {
    hash_u64(hash, value.size());
    for (const char character : value) {
        hash ^= static_cast<unsigned char>(character);
        hash *= fnv_prime;
    }
}

[[nodiscard]] std::uint64_t stable_string_hash(
    const std::string_view value
) noexcept {
    std::uint64_t hash = fnv_offset_basis;
    hash_string(hash, value);
    return hash;
}

void hash_float(std::uint64_t& hash, const float value) noexcept {
    hash_u64(hash, std::bit_cast<std::uint32_t>(value));
}

[[nodiscard]] std::uint16_t read_u16_le(
    const std::span<const std::uint8_t> bytes,
    const std::size_t offset
) {
    if (offset + 2U > bytes.size()) {
        throw std::runtime_error("truncated BMP header");
    }
    return static_cast<std::uint16_t>(bytes[offset]) |
        static_cast<std::uint16_t>(static_cast<std::uint16_t>(bytes[offset + 1U]) << 8U);
}

[[nodiscard]] std::uint32_t read_u32_le(
    const std::span<const std::uint8_t> bytes,
    const std::size_t offset
) {
    if (offset + 4U > bytes.size()) {
        throw std::runtime_error("truncated BMP header");
    }
    return static_cast<std::uint32_t>(bytes[offset]) |
        (static_cast<std::uint32_t>(bytes[offset + 1U]) << 8U) |
        (static_cast<std::uint32_t>(bytes[offset + 2U]) << 16U) |
        (static_cast<std::uint32_t>(bytes[offset + 3U]) << 24U);
}

[[nodiscard]] std::int32_t read_i32_le(
    const std::span<const std::uint8_t> bytes,
    const std::size_t offset
) {
    return std::bit_cast<std::int32_t>(read_u32_le(bytes, offset));
}

void validate_dimensions(
    const std::size_t width,
    const std::size_t height,
    const ImageLimits& limits
) {
    if (width == 0U || height == 0U || width > limits.maximum_width ||
        height > limits.maximum_height ||
        width > std::numeric_limits<std::size_t>::max() / height ||
        width * height > limits.maximum_pixels ||
        width * height > std::numeric_limits<std::size_t>::max() / 3U) {
        throw std::runtime_error("image dimensions exceed configured limits");
    }
}

[[nodiscard]] std::vector<std::uint8_t> read_file_bytes(
    const std::filesystem::path& path,
    const std::size_t maximum_bytes
) {
    std::ifstream input(path, std::ios::binary | std::ios::ate);
    if (!input) {
        throw std::runtime_error("unable to open image: " + path.string());
    }
    const std::streampos end = input.tellg();
    if (end < 0) {
        throw std::runtime_error("unable to determine image size: " + path.string());
    }
    const auto file_size = static_cast<std::uint64_t>(end);
    if (file_size > maximum_bytes ||
        file_size > static_cast<std::uint64_t>(std::numeric_limits<std::size_t>::max())) {
        throw std::runtime_error("image file exceeds configured byte limit");
    }
    std::vector<std::uint8_t> bytes(static_cast<std::size_t>(file_size));
    input.seekg(0, std::ios::beg);
    if (!bytes.empty()) {
        input.read(
            reinterpret_cast<char*>(bytes.data()),
            static_cast<std::streamsize>(bytes.size())
        );
    }
    if (!input) {
        throw std::runtime_error("failed while reading image: " + path.string());
    }
    return bytes;
}

[[nodiscard]] std::size_t skip_pnm_space(
    const std::span<const std::uint8_t> bytes,
    std::size_t position
) {
    while (position < bytes.size()) {
        if (bytes[position] == static_cast<std::uint8_t>('#')) {
            while (position < bytes.size() && bytes[position] != static_cast<std::uint8_t>('\n')) {
                ++position;
            }
        } else if (std::isspace(bytes[position]) != 0) {
            ++position;
        } else {
            break;
        }
    }
    return position;
}

[[nodiscard]] std::pair<std::size_t, std::size_t> parse_pnm_number(
    const std::span<const std::uint8_t> bytes,
    std::size_t position
) {
    position = skip_pnm_space(bytes, position);
    if (position >= bytes.size() || std::isdigit(bytes[position]) == 0) {
        throw std::runtime_error("invalid PNM header");
    }
    std::size_t value = 0U;
    while (position < bytes.size() && std::isdigit(bytes[position]) != 0) {
        const std::size_t digit = bytes[position] - static_cast<std::uint8_t>('0');
        if (value > (std::numeric_limits<std::size_t>::max() - digit) / 10U) {
            throw std::runtime_error("PNM header integer overflow");
        }
        value = value * 10U + digit;
        ++position;
    }
    return {value, position};
}

[[nodiscard]] ImageData load_pnm(
    const std::span<const std::uint8_t> bytes,
    const ImageLimits& limits
) {
    if (bytes.size() < 3U || bytes[0] != static_cast<std::uint8_t>('P') ||
        (bytes[1] != static_cast<std::uint8_t>('5') &&
         bytes[1] != static_cast<std::uint8_t>('6'))) {
        throw std::runtime_error("unsupported PNM format; expected binary P5 or P6");
    }
    const bool rgb = bytes[1] == static_cast<std::uint8_t>('6');
    auto [width, position] = parse_pnm_number(bytes, 2U);
    auto [height, after_height] = parse_pnm_number(bytes, position);
    auto [maximum, after_maximum] = parse_pnm_number(bytes, after_height);
    if (maximum == 0U || maximum > 255U) {
        throw std::runtime_error("Solstice supports 8-bit PNM images only");
    }
    validate_dimensions(width, height, limits);
    position = after_maximum;
    if (position >= bytes.size() || std::isspace(bytes[position]) == 0) {
        throw std::runtime_error("invalid PNM data separator");
    }
    ++position;
    const std::size_t channels = rgb ? 3U : 1U;
    const std::size_t source_bytes = width * height * channels;
    if (position + source_bytes > bytes.size()) {
        throw std::runtime_error("truncated PNM pixel payload");
    }
    ImageData image;
    image.width = width;
    image.height = height;
    image.rgb = std::vector<std::uint8_t>(width * height * 3U);
    for (std::size_t pixel = 0U; pixel < width * height; ++pixel) {
        if (rgb) {
            image.rgb[pixel * 3U] = bytes[position + pixel * 3U];
            image.rgb[pixel * 3U + 1U] = bytes[position + pixel * 3U + 1U];
            image.rgb[pixel * 3U + 2U] = bytes[position + pixel * 3U + 2U];
        } else {
            const std::uint8_t value = bytes[position + pixel];
            image.rgb[pixel * 3U] = value;
            image.rgb[pixel * 3U + 1U] = value;
            image.rgb[pixel * 3U + 2U] = value;
        }
    }
    return image;
}

[[nodiscard]] ImageData load_bmp(
    const std::span<const std::uint8_t> bytes,
    const ImageLimits& limits
) {
    if (bytes.size() < 54U || bytes[0] != static_cast<std::uint8_t>('B') ||
        bytes[1] != static_cast<std::uint8_t>('M')) {
        throw std::runtime_error("invalid BMP file");
    }
    const std::uint32_t pixel_offset = read_u32_le(bytes, 10U);
    const std::uint32_t header_size = read_u32_le(bytes, 14U);
    const std::int32_t signed_width = read_i32_le(bytes, 18U);
    const std::int32_t signed_height = read_i32_le(bytes, 22U);
    const std::uint16_t planes = read_u16_le(bytes, 26U);
    const std::uint16_t bits_per_pixel = read_u16_le(bytes, 28U);
    const std::uint32_t compression = read_u32_le(bytes, 30U);
    if (header_size < 40U || signed_width <= 0 || signed_height == 0 ||
        signed_height == std::numeric_limits<std::int32_t>::min() ||
        planes != 1U || (bits_per_pixel != 24U && bits_per_pixel != 32U) ||
        compression != 0U) {
        throw std::runtime_error("unsupported BMP; expected uncompressed 24-bit or 32-bit RGB");
    }
    const std::size_t width = static_cast<std::size_t>(signed_width);
    const std::size_t height = static_cast<std::size_t>(
        signed_height < 0 ? -signed_height : signed_height
    );
    validate_dimensions(width, height, limits);
    const std::size_t bytes_per_pixel = bits_per_pixel / 8U;
    const std::size_t row_unpadded = width * bytes_per_pixel;
    const std::size_t row_stride = (row_unpadded + 3U) & ~std::size_t{3U};
    if (pixel_offset > bytes.size() ||
        row_stride > (bytes.size() - pixel_offset) / height) {
        throw std::runtime_error("truncated BMP pixel payload");
    }
    ImageData image;
    image.width = width;
    image.height = height;
    image.rgb = std::vector<std::uint8_t>(width * height * 3U);
    const bool top_down = signed_height < 0;
    for (std::size_t output_y = 0U; output_y < height; ++output_y) {
        const std::size_t source_y = top_down ? output_y : height - 1U - output_y;
        const std::size_t row = static_cast<std::size_t>(pixel_offset) + source_y * row_stride;
        for (std::size_t x = 0U; x < width; ++x) {
            const std::size_t source = row + x * bytes_per_pixel;
            const std::size_t target = (output_y * width + x) * 3U;
            image.rgb[target] = bytes[source + 2U];
            image.rgb[target + 1U] = bytes[source + 1U];
            image.rgb[target + 2U] = bytes[source];
        }
    }
    return image;
}

#if defined(RLF_HAS_LIBPNG)
[[nodiscard]] ImageData load_png(
    const std::span<const std::uint8_t> bytes,
    const ImageLimits& limits
) {
    png_image decoder{};
    decoder.version = PNG_IMAGE_VERSION;
    if (png_image_begin_read_from_memory(&decoder, bytes.data(), bytes.size()) == 0) {
        throw std::runtime_error(
            "libpng could not read image: " + std::string(decoder.message)
        );
    }
    try {
        validate_dimensions(decoder.width, decoder.height, limits);
        decoder.format = PNG_FORMAT_RGB;
        ImageData image;
        image.width = decoder.width;
        image.height = decoder.height;
        image.rgb = std::vector<std::uint8_t>(PNG_IMAGE_SIZE(decoder));
        if (png_image_finish_read(
                &decoder, nullptr, image.rgb.data(), 0, nullptr
            ) == 0) {
            throw std::runtime_error(
                "libpng could not decode image: " + std::string(decoder.message)
            );
        }
        png_image_free(&decoder);
        return image;
    } catch (...) {
        png_image_free(&decoder);
        throw;
    }
}
#endif

#if defined(RLF_HAS_LIBJPEG)
struct JpegErrorManager final {
    jpeg_error_mgr base{};
    jmp_buf jump{};
    std::array<char, JMSG_LENGTH_MAX> message{};
};

extern "C" void solstice_jpeg_error_exit(j_common_ptr common) {
    auto* error = reinterpret_cast<JpegErrorManager*>(common->err);
    error->base.format_message(common, error->message.data());
    longjmp(error->jump, 1);
}

[[nodiscard]] ImageData load_jpeg(
    const std::span<const std::uint8_t> bytes,
    const ImageLimits& limits
) {
    if (bytes.size() > static_cast<std::size_t>(
            std::numeric_limits<unsigned long>::max()
        )) {
        throw std::runtime_error("JPEG input exceeds decoder address range");
    }

    jpeg_decompress_struct decoder{};
    JpegErrorManager error{};
    decoder.err = jpeg_std_error(&error.base);
    error.base.error_exit = solstice_jpeg_error_exit;
    if (setjmp(error.jump) != 0) {
        jpeg_destroy_decompress(&decoder);
        throw std::runtime_error(
            "libjpeg could not decode image: " + std::string(error.message.data())
        );
    }

    jpeg_create_decompress(&decoder);
    jpeg_mem_src(
        &decoder,
        bytes.data(),
        static_cast<unsigned long>(bytes.size())
    );
    if (jpeg_read_header(&decoder, TRUE) != JPEG_HEADER_OK) {
        jpeg_destroy_decompress(&decoder);
        throw std::runtime_error("invalid JPEG header");
    }
    decoder.out_color_space = JCS_RGB;
    static_cast<void>(jpeg_start_decompress(&decoder));
    validate_dimensions(decoder.output_width, decoder.output_height, limits);

    ImageData image;
    image.width = decoder.output_width;
    image.height = decoder.output_height;
    image.rgb = std::vector<std::uint8_t>(image.width * image.height * 3U);
    const std::size_t stride = image.width * 3U;
    while (decoder.output_scanline < decoder.output_height) {
        JSAMPROW row = image.rgb.data() +
            static_cast<std::size_t>(decoder.output_scanline) * stride;
        static_cast<void>(jpeg_read_scanlines(&decoder, &row, 1U));
    }
    static_cast<void>(jpeg_finish_decompress(&decoder));
    jpeg_destroy_decompress(&decoder);
    return image;
}
#endif

#if defined(_WIN32)
class ComInitialization final {
public:
    ComInitialization() {
        const HRESULT result = CoInitializeEx(nullptr, COINIT_MULTITHREADED);
        should_uninitialize_ = SUCCEEDED(result);
        if (FAILED(result) && result != RPC_E_CHANGED_MODE) {
            throw std::runtime_error("unable to initialize Windows Imaging Component");
        }
    }

    ~ComInitialization() {
        if (should_uninitialize_) {
            CoUninitialize();
        }
    }

    ComInitialization(const ComInitialization&) = delete;
    ComInitialization& operator=(const ComInitialization&) = delete;

private:
    bool should_uninitialize_{};
};

class ComRelease final {
public:
    template <typename Type>
    void operator()(Type* pointer) const noexcept {
        if (pointer != nullptr) {
            pointer->Release();
        }
    }
};

template <typename Type>
using ComPointer = std::unique_ptr<Type, ComRelease>;

[[nodiscard]] ImageData load_wic(
    const std::filesystem::path& path,
    const ImageLimits& limits
) {
    const ComInitialization com_initialization;
    static_cast<void>(com_initialization);
    IWICImagingFactory* factory_raw = nullptr;
    const HRESULT factory_result = CoCreateInstance(
        CLSID_WICImagingFactory,
        nullptr,
        CLSCTX_INPROC_SERVER,
        IID_PPV_ARGS(&factory_raw)
    );
    ComPointer<IWICImagingFactory> factory(factory_raw);
    if (FAILED(factory_result)) {
        throw std::runtime_error("unable to create Windows Imaging Component factory");
    }
    IWICBitmapDecoder* decoder_raw = nullptr;
    const HRESULT decoder_result = factory->CreateDecoderFromFilename(
        path.c_str(), nullptr, GENERIC_READ, WICDecodeMetadataCacheOnLoad, &decoder_raw
    );
    ComPointer<IWICBitmapDecoder> decoder(decoder_raw);
    if (FAILED(decoder_result)) {
        throw std::runtime_error("Windows Imaging Component could not decode the image");
    }
    IWICBitmapFrameDecode* frame_raw = nullptr;
    const HRESULT frame_result = decoder->GetFrame(0U, &frame_raw);
    ComPointer<IWICBitmapFrameDecode> frame(frame_raw);
    if (FAILED(frame_result)) {
        throw std::runtime_error("unable to read image frame");
    }
    UINT width = 0U;
    UINT height = 0U;
    if (FAILED(frame->GetSize(&width, &height))) {
        throw std::runtime_error("unable to read image dimensions");
    }
    validate_dimensions(width, height, limits);
    IWICFormatConverter* converter_raw = nullptr;
    const HRESULT converter_result = factory->CreateFormatConverter(&converter_raw);
    ComPointer<IWICFormatConverter> converter(converter_raw);
    if (FAILED(converter_result) || FAILED(converter->Initialize(
        frame.get(), GUID_WICPixelFormat24bppRGB, WICBitmapDitherTypeNone,
        nullptr, 0.0, WICBitmapPaletteTypeCustom
    ))) {
        throw std::runtime_error("unable to convert image to RGB");
    }
    ImageData image;
    image.width = width;
    image.height = height;
    image.rgb = std::vector<std::uint8_t>(image.width * image.height * 3U);
    const UINT stride = width * 3U;
    const UINT buffer_size = static_cast<UINT>(image.rgb.size());
    if (FAILED(converter->CopyPixels(
        nullptr, stride, buffer_size, image.rgb.data()
    ))) {
        throw std::runtime_error("unable to copy decoded image pixels");
    }
    return image;
}
#endif

[[nodiscard]] std::string lowercase_extension(const std::filesystem::path& path) {
    std::string extension = path.extension().string();
    std::transform(
        extension.begin(), extension.end(), extension.begin(),
        [](const unsigned char character) {
            return static_cast<char>(std::tolower(character));
        }
    );
    return extension;
}

[[nodiscard]] std::string top_concept(const VisualMode& mode) {
    if (mode.concepts.empty()) {
        return {};
    }
    const auto found = std::max_element(
        mode.concepts.begin(), mode.concepts.end(),
        [](const VisualConceptCount& left, const VisualConceptCount& right) {
            if (left.count != right.count) {
                return left.count < right.count;
            }
            return left.concept_name > right.concept_name;
        }
    );
    return found->concept_name;
}

}  // namespace

ImageData decode_image(
    const std::span<const std::uint8_t> encoded,
    const std::string_view extension,
    const ImageLimits limits
) {
    std::string normalized(extension);
    std::transform(
        normalized.begin(), normalized.end(), normalized.begin(),
        [](const unsigned char character) {
            return static_cast<char>(std::tolower(character));
        }
    );
    if (normalized == ".pnm" || normalized == ".ppm" || normalized == ".pgm") {
        return load_pnm(encoded, limits);
    }
    if (normalized == ".bmp" || normalized == ".dib") {
        return load_bmp(encoded, limits);
    }
#if defined(RLF_HAS_LIBPNG)
    if (normalized == ".png") return load_png(encoded, limits);
#endif
#if defined(RLF_HAS_LIBJPEG)
    if (normalized == ".jpg" || normalized == ".jpeg" || normalized == ".jpe") {
        return load_jpeg(encoded, limits);
    }
#endif
    throw std::runtime_error(
        "unsupported encoded image format; enable libpng/libjpeg or use PPM/PGM/BMP"
    );
}

ImageData load_image(const std::filesystem::path& path, const ImageLimits limits) {
    const std::string extension = lowercase_extension(path);
    if (extension == ".pnm" || extension == ".ppm" || extension == ".pgm" ||
        extension == ".bmp" || extension == ".dib"
#if defined(RLF_HAS_LIBPNG)
        || extension == ".png"
#endif
#if defined(RLF_HAS_LIBJPEG)
        || extension == ".jpg" || extension == ".jpeg" || extension == ".jpe"
#endif
    ) {
        return decode_image(
            read_file_bytes(path, limits.maximum_file_bytes), extension, limits
        );
    }
#if defined(_WIN32)
    const auto file_size = std::filesystem::file_size(path);
    if (file_size > limits.maximum_file_bytes) {
        throw std::runtime_error("image file exceeds configured byte limit");
    }
    return load_wic(path, limits);
#else
    throw std::runtime_error(
        "unsupported image format on this build; enable libpng/libjpeg or use "
        "PPM/PGM/BMP"
    );
#endif
}

VisualPatchFabric::VisualPatchFabric(VisionConfig config)
    : config_(config),
      backend_(rlf::frontier::make_frontier_backend(
          rlf::frontier::FrontierBackendKind::optimized_cpu
      )),
      mode_router_(config_.sparse_router),
      incremental_sparse_router_updates_(
          incremental_sparse_router_updates_from_environment()
      ),
      batched_sparse_reranking_(batched_sparse_reranking_from_environment()),
      sparse_rerank_policy_explicit_(
          sparse_rerank_policy_explicit_from_environment()
      ),
      indexed_concept_updates_(indexed_concept_updates_from_environment()),
      indexed_example_duplicate_lookup_(
          indexed_example_duplicate_lookup_from_environment()
      ),
      persistent_mode_id_index_(persistent_mode_id_index_from_environment()) {
    if (config_.patch_size == 0U ||
        (config_.descriptor_dimensions != legacy_descriptor_size &&
         config_.descriptor_dimensions != frontier_descriptor_size) ||
        config_.maximum_input_side == 0U || config_.maximum_patches == 0U ||
        config_.retrieval_query_batch == 0U ||
        config_.retrieval_candidate_batch == 0U ||
        config_.training_patch_batch == 0U ||
        config_.sparse_routing_minimum_modes == 0U ||
        config_.maximum_modes == 0U || config_.maximum_examples == 0U ||
        config_.maximum_regions == 0U ||
        config_.maximum_concepts_per_mode == 0U ||
        std::any_of(
            config_.patch_sizes.begin(), config_.patch_sizes.end(),
            [](const std::size_t value) { return value == 0U; }
        ) ||
        !std::isfinite(config_.mode_creation_similarity) ||
        config_.mode_creation_similarity < 0.0 ||
        config_.mode_creation_similarity > 1.0 ||
        !std::isfinite(config_.example_match_similarity) ||
        config_.example_match_similarity < 0.0 ||
        config_.example_match_similarity > 1.0 ||
        !std::isfinite(config_.local_learning_rate) ||
        config_.local_learning_rate <= 0.0 || config_.local_learning_rate > 1.0) {
        throw std::invalid_argument("invalid Solstice vision configuration");
    }
}

VisualPatchFabric::PreparedImage VisualPatchFabric::prepare_image(
    const ImageData& image
) const {
    if (image.width == 0U || image.height == 0U ||
        image.width > std::numeric_limits<std::size_t>::max() / image.height ||
        image.rgb.size() != image.width * image.height * 3U) {
        throw std::invalid_argument("invalid RGB image supplied to Solstice vision fabric");
    }

    const std::size_t largest_side = std::max(image.width, image.height);
    double scale = largest_side > config_.maximum_input_side
        ? static_cast<double>(config_.maximum_input_side) /
            static_cast<double>(largest_side)
        : 1.0;

    std::vector<std::size_t> patch_sizes = config_.patch_sizes;
    if (patch_sizes.empty()) {
        patch_sizes.push_back(config_.patch_size);
    }
    std::sort(patch_sizes.begin(), patch_sizes.end());
    patch_sizes.erase(std::unique(patch_sizes.begin(), patch_sizes.end()), patch_sizes.end());

    const auto patch_count_for = [&patch_sizes, &image](const double candidate_scale) {
        const std::size_t width = std::max<std::size_t>(
            1U,
            static_cast<std::size_t>(
                std::floor(static_cast<double>(image.width) * candidate_scale + 0.5)
            )
        );
        const std::size_t height = std::max<std::size_t>(
            1U,
            static_cast<std::size_t>(
                std::floor(static_cast<double>(image.height) * candidate_scale + 0.5)
            )
        );
        std::size_t count = 0U;
        for (const std::size_t patch_size : patch_sizes) {
            const std::size_t grid_width = (width + patch_size - 1U) / patch_size;
            const std::size_t grid_height = (height + patch_size - 1U) / patch_size;
            if (grid_width > std::numeric_limits<std::size_t>::max() / grid_height ||
                count > std::numeric_limits<std::size_t>::max() - grid_width * grid_height) {
                return std::numeric_limits<std::size_t>::max();
            }
            count += grid_width * grid_height;
        }
        return count;
    };

    if (patch_count_for(scale) > config_.maximum_patches) {
        double low = 0.0;
        double high = scale;
        for (std::size_t iteration = 0U; iteration < 32U; ++iteration) {
            const double middle = (low + high) * 0.5;
            if (patch_count_for(middle) <= config_.maximum_patches) {
                low = middle;
            } else {
                high = middle;
            }
        }
        scale = std::max(low, 1.0 / static_cast<double>(largest_side));
    }

    const std::size_t target_width = std::max<std::size_t>(
        1U,
        static_cast<std::size_t>(
            std::floor(static_cast<double>(image.width) * scale + 0.5)
        )
    );
    const std::size_t target_height = std::max<std::size_t>(
        1U,
        static_cast<std::size_t>(
            std::floor(static_cast<double>(image.height) * scale + 0.5)
        )
    );
    if (target_width == image.width && target_height == image.height) {
        return PreparedImage{image, 1.0, 1.0};
    }

    ImageData resized;
    resized.width = target_width;
    resized.height = target_height;
    resized.rgb = std::vector<std::uint8_t>(target_width * target_height * 3U);
    const double x_ratio = static_cast<double>(image.width) /
        static_cast<double>(target_width);
    const double y_ratio = static_cast<double>(image.height) /
        static_cast<double>(target_height);
    for (std::size_t y = 0U; y < target_height; ++y) {
        const double source_y = (static_cast<double>(y) + 0.5) * y_ratio - 0.5;
        const std::size_t y0 = static_cast<std::size_t>(std::clamp(
            std::floor(source_y), 0.0, static_cast<double>(image.height - 1U)
        ));
        const std::size_t y1 = std::min(y0 + 1U, image.height - 1U);
        const double wy = std::clamp(source_y - std::floor(source_y), 0.0, 1.0);
        for (std::size_t x = 0U; x < target_width; ++x) {
            const double source_x = (static_cast<double>(x) + 0.5) * x_ratio - 0.5;
            const std::size_t x0 = static_cast<std::size_t>(std::clamp(
                std::floor(source_x), 0.0, static_cast<double>(image.width - 1U)
            ));
            const std::size_t x1 = std::min(x0 + 1U, image.width - 1U);
            const double wx = std::clamp(source_x - std::floor(source_x), 0.0, 1.0);
            for (std::size_t channel = 0U; channel < 3U; ++channel) {
                const double top =
                    (1.0 - wx) * image.rgb[(y0 * image.width + x0) * 3U + channel] +
                    wx * image.rgb[(y0 * image.width + x1) * 3U + channel];
                const double bottom =
                    (1.0 - wx) * image.rgb[(y1 * image.width + x0) * 3U + channel] +
                    wx * image.rgb[(y1 * image.width + x1) * 3U + channel];
                const double value = (1.0 - wy) * top + wy * bottom;
                resized.rgb[(y * target_width + x) * 3U + channel] =
                    static_cast<std::uint8_t>(std::clamp(value, 0.0, 255.0) + 0.5);
            }
        }
    }
    return PreparedImage{
        std::move(resized),
        static_cast<double>(image.width) / static_cast<double>(target_width),
        static_cast<double>(image.height) / static_cast<double>(target_height),
    };
}

std::vector<VisualPatchFabric::PatchRecord> VisualPatchFabric::extract_patches(
    const ImageData& image
) const {
    if (image.width == 0U || image.height == 0U ||
        image.rgb.size() != image.width * image.height * 3U) {
        throw std::invalid_argument("invalid RGB image supplied to Solstice vision fabric");
    }

    std::vector<std::size_t> patch_sizes = config_.patch_sizes;
    if (patch_sizes.empty()) {
        patch_sizes.push_back(config_.patch_size);
    }
    std::sort(patch_sizes.begin(), patch_sizes.end());
    patch_sizes.erase(std::unique(patch_sizes.begin(), patch_sizes.end()), patch_sizes.end());

    std::size_t estimated_count = 0U;
    for (const std::size_t patch_size : patch_sizes) {
        const std::size_t grid_width = (image.width + patch_size - 1U) / patch_size;
        const std::size_t grid_height = (image.height + patch_size - 1U) / patch_size;
        if (grid_width > std::numeric_limits<std::size_t>::max() / grid_height ||
            estimated_count > std::numeric_limits<std::size_t>::max() - grid_width * grid_height) {
            throw std::runtime_error("visual patch count exceeds addressable memory");
        }
        estimated_count += grid_width * grid_height;
    }

    std::vector<PatchRecord> patches;
    patches.reserve(estimated_count);
    for (std::size_t scale_index = 0U; scale_index < patch_sizes.size(); ++scale_index) {
        const std::size_t patch_size = patch_sizes[scale_index];
        const std::size_t grid_width = (image.width + patch_size - 1U) / patch_size;
        const std::size_t grid_height = (image.height + patch_size - 1U) / patch_size;
        for (std::size_t grid_y = 0U; grid_y < grid_height; ++grid_y) {
            for (std::size_t grid_x = 0U; grid_x < grid_width; ++grid_x) {
                const std::size_t start_x = grid_x * patch_size;
                const std::size_t start_y = grid_y * patch_size;
                const std::size_t patch_width = std::min(patch_size, image.width - start_x);
                const std::size_t patch_height = std::min(patch_size, image.height - start_y);
                std::array<double, 3U> sum{};
                std::array<double, 3U> squared{};
                std::array<double, 8U> luma_histogram{};
                std::array<double, 4U> orientation_histogram{};
                double gradient_x = 0.0;
                double gradient_y = 0.0;
                double diagonal_gradient = 0.0;
                double edge_count = 0.0;
                double saturation_sum = 0.0;
                double luma_sum = 0.0;
                double luma_squared = 0.0;
                double minimum_luma = 1.0;
                double maximum_luma = 0.0;
                double center_luma = 0.0;
                double border_luma = 0.0;
                double center_count = 0.0;
                double border_count = 0.0;
                double red_green_sum = 0.0;
                double red_green_squared = 0.0;
                double blue_yellow_sum = 0.0;
                double blue_yellow_squared = 0.0;
                for (std::size_t local_y = 0U; local_y < patch_height; ++local_y) {
                    for (std::size_t local_x = 0U; local_x < patch_width; ++local_x) {
                        const std::size_t x = start_x + local_x;
                        const std::size_t y = start_y + local_y;
                        const std::size_t offset = (y * image.width + x) * 3U;
                        const double red = static_cast<double>(image.rgb[offset]) / 255.0;
                        const double green = static_cast<double>(image.rgb[offset + 1U]) / 255.0;
                        const double blue = static_cast<double>(image.rgb[offset + 2U]) / 255.0;
                        const std::array<double, 3U> values{red, green, blue};
                        for (std::size_t channel = 0U; channel < 3U; ++channel) {
                            sum[channel] += values[channel];
                            squared[channel] += values[channel] * values[channel];
                        }
                        const double maximum = std::max({red, green, blue});
                        const double minimum = std::min({red, green, blue});
                        saturation_sum += maximum <= 1.0e-9
                            ? 0.0
                            : (maximum - minimum) / maximum;
                        const double luma = 0.299 * red + 0.587 * green + 0.114 * blue;
                        luma_sum += luma;
                        luma_squared += luma * luma;
                        minimum_luma = std::min(minimum_luma, luma);
                        maximum_luma = std::max(maximum_luma, luma);
                        const std::size_t luma_bin = std::min<std::size_t>(
                            7U,
                            static_cast<std::size_t>(luma * 8.0)
                        );
                        luma_histogram[luma_bin] += 1.0;
                        const bool center =
                            local_x * 4U >= patch_width && local_x * 4U < patch_width * 3U &&
                            local_y * 4U >= patch_height && local_y * 4U < patch_height * 3U;
                        if (center) {
                            center_luma += luma;
                            center_count += 1.0;
                        } else {
                            border_luma += luma;
                            border_count += 1.0;
                        }
                        const double red_green = red - green;
                        const double blue_yellow = blue - 0.5 * (red + green);
                        red_green_sum += red_green;
                        red_green_squared += red_green * red_green;
                        blue_yellow_sum += blue_yellow;
                        blue_yellow_squared += blue_yellow * blue_yellow;

                        double signed_x = 0.0;
                        double signed_y = 0.0;
                        if (x + 1U < image.width) {
                            const std::size_t next = (y * image.width + x + 1U) * 3U;
                            const double next_luma = (
                                0.299 * image.rgb[next] +
                                0.587 * image.rgb[next + 1U] +
                                0.114 * image.rgb[next + 2U]
                            ) / 255.0;
                            signed_x = next_luma - luma;
                            gradient_x += std::abs(signed_x);
                            edge_count += std::abs(signed_x) > 0.12 ? 1.0 : 0.0;
                        }
                        if (y + 1U < image.height) {
                            const std::size_t next = ((y + 1U) * image.width + x) * 3U;
                            const double next_luma = (
                                0.299 * image.rgb[next] +
                                0.587 * image.rgb[next + 1U] +
                                0.114 * image.rgb[next + 2U]
                            ) / 255.0;
                            signed_y = next_luma - luma;
                            gradient_y += std::abs(signed_y);
                            edge_count += std::abs(signed_y) > 0.12 ? 1.0 : 0.0;
                        }
                        const double magnitude = std::hypot(signed_x, signed_y);
                        if (magnitude > 1.0e-6) {
                            double angle = std::atan2(signed_y, signed_x);
                            if (angle < 0.0) {
                                angle += 3.14159265358979323846;
                            }
                            if (angle >= 3.14159265358979323846) {
                                angle -= 3.14159265358979323846;
                            }
                            const std::size_t bin = std::min<std::size_t>(
                                3U,
                                static_cast<std::size_t>(
                                    angle * 4.0 / 3.14159265358979323846
                                )
                            );
                            orientation_histogram[bin] += magnitude;
                        }
                        if (x + 1U < image.width && y + 1U < image.height) {
                            const std::size_t next = ((y + 1U) * image.width + x + 1U) * 3U;
                            const double next_luma = (
                                0.299 * image.rgb[next] +
                                0.587 * image.rgb[next + 1U] +
                                0.114 * image.rgb[next + 2U]
                            ) / 255.0;
                            diagonal_gradient += std::abs(next_luma - luma);
                        }
                    }
                }
                const double count = static_cast<double>(patch_width * patch_height);
                std::vector<float> descriptor(config_.descriptor_dimensions, 0.0F);
                for (std::size_t channel = 0U; channel < 3U; ++channel) {
                    const double mean = sum[channel] / count;
                    descriptor[channel] = static_cast<float>(mean);
                    descriptor[channel + 3U] = static_cast<float>(
                        std::sqrt(std::max(0.0, squared[channel] / count - mean * mean))
                    );
                }
                descriptor[6U] = static_cast<float>(luma_sum / count);
                descriptor[7U] = static_cast<float>(gradient_x / count);
                descriptor[8U] = static_cast<float>(gradient_y / count);
                descriptor[9U] = static_cast<float>(edge_count / (2.0 * count));
                descriptor[10U] = static_cast<float>(saturation_sum / count);
                descriptor[11U] = static_cast<float>(
                    (static_cast<double>(start_x) + static_cast<double>(patch_width) * 0.5) /
                    static_cast<double>(image.width)
                );
                descriptor[12U] = static_cast<float>(
                    (static_cast<double>(start_y) + static_cast<double>(patch_height) * 0.5) /
                    static_cast<double>(image.height)
                );
                descriptor[13U] = static_cast<float>(
                    static_cast<double>(patch_width) / static_cast<double>(image.width)
                );
                descriptor[14U] = static_cast<float>(
                    static_cast<double>(patch_height) / static_cast<double>(image.height)
                );
                descriptor[15U] = static_cast<float>(diagonal_gradient / count);
                if (descriptor.size() >= frontier_descriptor_size) {
                    for (std::size_t bin = 0U; bin < 4U; ++bin) {
                        descriptor[16U + bin] = static_cast<float>(
                            (luma_histogram[bin * 2U] + luma_histogram[bin * 2U + 1U]) /
                            count
                        );
                    }
                    const double orientation_total = std::max(
                        1.0e-12,
                        orientation_histogram[0U] + orientation_histogram[1U] +
                        orientation_histogram[2U] + orientation_histogram[3U]
                    );
                    for (std::size_t bin = 0U; bin < 4U; ++bin) {
                        descriptor[20U + bin] = static_cast<float>(
                            orientation_histogram[bin] / orientation_total
                        );
                    }
                    descriptor[24U] = static_cast<float>(red_green_sum / count);
                    descriptor[25U] = static_cast<float>(blue_yellow_sum / count);
                    descriptor[26U] = static_cast<float>(maximum_luma - minimum_luma);
                    double entropy = 0.0;
                    for (const double frequency : luma_histogram) {
                        if (frequency > 0.0) {
                            const double probability = frequency / count;
                            entropy -= probability * std::log2(probability);
                        }
                    }
                    descriptor[27U] = static_cast<float>(entropy / 3.0);
                    const double center_mean = center_count > 0.0
                        ? center_luma / center_count
                        : luma_sum / count;
                    const double border_mean = border_count > 0.0
                        ? border_luma / border_count
                        : luma_sum / count;
                    descriptor[28U] = static_cast<float>(center_mean - border_mean);
                    const double rg_mean = red_green_sum / count;
                    const double by_mean = blue_yellow_sum / count;
                    const double rg_variance = std::max(
                        0.0,
                        red_green_squared / count - rg_mean * rg_mean
                    );
                    const double by_variance = std::max(
                        0.0,
                        blue_yellow_squared / count - by_mean * by_mean
                    );
                    descriptor[29U] = static_cast<float>(
                        std::sqrt(rg_variance + by_variance)
                    );
                    descriptor[30U] = static_cast<float>(
                        static_cast<double>(patch_size) /
                        static_cast<double>(std::max(image.width, image.height))
                    );
                    descriptor[31U] = static_cast<float>(
                        static_cast<double>(std::min(patch_width, patch_height)) /
                        static_cast<double>(std::max(patch_width, patch_height))
                    );
                }
                patches.push_back(PatchRecord{
                    scale_index, patch_size, grid_x, grid_y, start_x, start_y,
                    patch_width, patch_height, std::move(descriptor), 0U, 0.0,
                });
            }
        }
    }
    return patches;
}

std::vector<float> VisualPatchFabric::global_descriptor(
    const ImageData& image,
    const std::span<const PatchRecord> patches
) const {
    std::vector<float> descriptor(config_.descriptor_dimensions, 0.0F);
    if (patches.empty()) {
        return descriptor;
    }
    for (const PatchRecord& patch : patches) {
        for (std::size_t index = 0U; index < descriptor.size(); ++index) {
            descriptor[index] += patch.descriptor[index];
        }
    }
    const float inverse = 1.0F / static_cast<float>(patches.size());
    for (float& value : descriptor) {
        value *= inverse;
    }
    descriptor[11U] = static_cast<float>(image.width) /
        static_cast<float>(std::max(image.width, image.height));
    descriptor[12U] = static_cast<float>(image.height) /
        static_cast<float>(std::max(image.width, image.height));
    return descriptor;
}

double VisualPatchFabric::descriptor_similarity(
    const std::span<const float> left,
    const std::span<const float> right
) noexcept {
    if (left.size() != right.size() || left.empty()) {
        return 0.0;
    }
    double squared_distance = 0.0;
    for (std::size_t index = 0U; index < left.size(); ++index) {
        const double difference = static_cast<double>(left[index]) - static_cast<double>(right[index]);
        squared_distance += difference * difference;
    }
    const double normalized = squared_distance / static_cast<double>(left.size());
    return std::clamp(std::exp(-8.0 * normalized), 0.0, 1.0);
}

void VisualPatchFabric::update_mode_router() const {
    if (mode_router_revision_ == mode_revision_) {
        return;
    }
    std::vector<float> matrix;
    matrix.reserve(modes_.size() * config_.descriptor_dimensions);
    for (const VisualMode& mode : modes_) {
        matrix.insert(matrix.end(), mode.prototype.begin(), mode.prototype.end());
    }
    mode_router_.rebuild(
        matrix, modes_.size(), config_.descriptor_dimensions
    );
    mode_router_revision_ = mode_revision_;
}

std::pair<std::size_t, double> VisualPatchFabric::nearest_mode(
    const std::span<const float> descriptor
) const {
    if (modes_.empty()) {
        return {0U, 0.0};
    }
    std::size_t best_index = 0U;
    double best_similarity = -1.0;
    if (modes_.size() >= config_.sparse_routing_minimum_modes) {
        update_mode_router();
        const SparseRouteResult route = mode_router_.route(descriptor);
        std::vector<float> candidates;
        candidates.reserve(route.candidate_indices.size() * descriptor.size());
        for (const std::size_t index : route.candidate_indices) {
            candidates.insert(
                candidates.end(),
                modes_[index].prototype.begin(),
                modes_[index].prototype.end()
            );
        }
        const std::vector<float> similarities = backend_->batch_cosine(
            descriptor, 1U, candidates, route.candidate_indices.size(),
            descriptor.size()
        );
        best_index = route.candidate_indices.front();
        for (std::size_t offset = 0U;
             offset < route.candidate_indices.size();
             ++offset) {
            const std::size_t index = route.candidate_indices[offset];
            const double similarity = std::clamp(
                (static_cast<double>(similarities[offset]) + 1.0) * 0.5,
                0.0, 1.0
            );
            if (similarity > best_similarity ||
                (similarity == best_similarity &&
                 modes_[index].id < modes_[best_index].id)) {
                best_index = index;
                best_similarity = similarity;
            }
        }
        return {best_index, std::max(0.0, best_similarity)};
    }
    for (std::size_t candidate_begin = 0U;
         candidate_begin < modes_.size();
         candidate_begin += config_.retrieval_candidate_batch) {
        const std::size_t candidate_count = std::min(
            config_.retrieval_candidate_batch,
            modes_.size() - candidate_begin
        );
        std::vector<float> candidates;
        candidates.reserve(candidate_count * descriptor.size());
        for (std::size_t offset = 0U; offset < candidate_count; ++offset) {
            const VisualMode& mode = modes_[candidate_begin + offset];
            candidates.insert(
                candidates.end(), mode.prototype.begin(), mode.prototype.end()
            );
        }
        const std::vector<float> similarities = backend_->batch_cosine(
            descriptor,
            1U,
            candidates,
            candidate_count,
            descriptor.size()
        );
        for (std::size_t offset = 0U; offset < candidate_count; ++offset) {
            const std::size_t index = candidate_begin + offset;
            const double similarity = std::clamp(
                (static_cast<double>(similarities[offset]) + 1.0) * 0.5,
                0.0,
                1.0
            );
            if (similarity > best_similarity ||
                (similarity == best_similarity &&
                 modes_[index].id < modes_[best_index].id)) {
                best_index = index;
                best_similarity = similarity;
            }
        }
    }
    return {best_index, std::max(0.0, best_similarity)};
}

void VisualPatchFabric::assign_existing_modes(
    const std::span<PatchRecord> patches
) const {
    if (patches.empty() || modes_.empty()) {
        return;
    }
    if (modes_.size() >= config_.sparse_routing_minimum_modes) {
        if (!batched_sparse_reranking_) {
            for (PatchRecord& patch : patches) {
                const auto nearest = nearest_mode(patch.descriptor);
                patch.mode_id = modes_[nearest.first].id;
                patch.mode_similarity = nearest.second;
            }
            return;
        }

        update_mode_router();
        const std::size_t dimension = config_.descriptor_dimensions;
        for (std::size_t query_begin = 0U;
             query_begin < patches.size();
             query_begin += config_.retrieval_query_batch) {
            const std::size_t query_count = std::min(
                config_.retrieval_query_batch, patches.size() - query_begin
            );
            std::vector<float> queries;
            queries.reserve(query_count * dimension);
            std::vector<float> candidates;
            std::vector<std::size_t> candidate_query_indices;
            std::vector<std::size_t> candidate_mode_indices;
            std::vector<std::size_t> route_offsets;
            route_offsets.reserve(query_count + 1U);
            route_offsets.push_back(0U);
            for (std::size_t query = 0U; query < query_count; ++query) {
                const PatchRecord& patch = patches[query_begin + query];
                queries.insert(
                    queries.end(), patch.descriptor.begin(), patch.descriptor.end()
                );
                const SparseRouteResult route = mode_router_.route(patch.descriptor);
                if (route.candidate_indices.empty()) {
                    throw std::logic_error("visual sparse router returned no candidates");
                }
                for (const std::size_t mode_index : route.candidate_indices) {
                    candidates.insert(
                        candidates.end(), modes_[mode_index].prototype.begin(),
                        modes_[mode_index].prototype.end()
                    );
                    candidate_query_indices.push_back(query);
                    candidate_mode_indices.push_back(mode_index);
                }
                route_offsets.push_back(candidate_mode_indices.size());
            }
            const std::vector<float> similarities = backend_->batch_cosine_indexed(
                queries, query_count, candidates, candidate_query_indices, dimension
            );
            for (std::size_t query = 0U; query < query_count; ++query) {
                const std::size_t begin = route_offsets[query];
                const std::size_t end = route_offsets[query + 1U];
                std::size_t best_index = candidate_mode_indices[begin];
                double best_similarity = -1.0;
                for (std::size_t pair = begin; pair < end; ++pair) {
                    const std::size_t mode_index = candidate_mode_indices[pair];
                    const double similarity = std::clamp(
                        (static_cast<double>(similarities[pair]) + 1.0) * 0.5,
                        0.0, 1.0
                    );
                    if (similarity > best_similarity ||
                        (similarity == best_similarity &&
                         modes_[mode_index].id < modes_[best_index].id)) {
                        best_index = mode_index;
                        best_similarity = similarity;
                    }
                }
                patches[query_begin + query].mode_id = modes_[best_index].id;
                patches[query_begin + query].mode_similarity =
                    std::max(0.0, best_similarity);
            }
        }
        return;
    }
    const std::size_t dimension = config_.descriptor_dimensions;
    std::vector<float> candidate_cache;
    candidate_cache.reserve(modes_.size() * dimension);
    for (const VisualMode& mode : modes_) {
        candidate_cache.insert(
            candidate_cache.end(), mode.prototype.begin(), mode.prototype.end()
        );
    }
    backend_->prepare_candidate_cache(
        candidate_cache,
        modes_.size(),
        dimension,
        mode_revision_
    );

    for (std::size_t query_begin = 0U;
         query_begin < patches.size();
         query_begin += config_.retrieval_query_batch) {
        const std::size_t query_count = std::min(
            config_.retrieval_query_batch,
            patches.size() - query_begin
        );
        std::vector<float> queries;
        queries.reserve(query_count * dimension);
        for (std::size_t offset = 0U; offset < query_count; ++offset) {
            const PatchRecord& patch = patches[query_begin + offset];
            queries.insert(
                queries.end(), patch.descriptor.begin(), patch.descriptor.end()
            );
        }

        std::vector<std::size_t> best_indices(query_count, 0U);
        std::vector<double> best_similarities(query_count, -1.0);
        for (std::size_t candidate_begin = 0U;
             candidate_begin < modes_.size();
             candidate_begin += config_.retrieval_candidate_batch) {
            const std::size_t candidate_count = std::min(
                config_.retrieval_candidate_batch,
                modes_.size() - candidate_begin
            );
            const std::vector<float> similarities = backend_->batch_cosine_cached(
                queries,
                query_count,
                candidate_begin,
                candidate_count,
                dimension,
                mode_revision_
            );
            for (std::size_t query = 0U; query < query_count; ++query) {
                for (std::size_t candidate = 0U;
                     candidate < candidate_count;
                     ++candidate) {
                    const std::size_t mode_index = candidate_begin + candidate;
                    const std::size_t result_index = query * candidate_count + candidate;
                    const double similarity = std::clamp(
                        (static_cast<double>(similarities[result_index]) + 1.0) * 0.5,
                        0.0,
                        1.0
                    );
                    if (similarity > best_similarities[query] ||
                        (similarity == best_similarities[query] &&
                         modes_[mode_index].id < modes_[best_indices[query]].id)) {
                        best_indices[query] = mode_index;
                        best_similarities[query] = similarity;
                    }
                }
            }
        }
        for (std::size_t query = 0U; query < query_count; ++query) {
            patches[query_begin + query].mode_id = modes_[best_indices[query]].id;
            patches[query_begin + query].mode_similarity =
                std::max(0.0, best_similarities[query]);
        }
    }
}

std::vector<std::string> VisualPatchFabric::caption_concepts(
    const std::string_view caption
) {
    static const std::unordered_set<std::string> stop_words{
        "a", "an", "the", "is", "are", "was", "were", "in", "on", "at",
        "to", "of", "and", "or", "with", "this", "that", "it", "image",
        "picture", "photo", "there", "shows", "showing", "from", "by",
    };
    std::vector<std::string> concepts;
    std::string word;
    const auto flush = [&concepts, &word]() {
        if (word.size() >= 2U && !stop_words.contains(word) &&
            std::find(concepts.begin(), concepts.end(), word) == concepts.end()) {
            concepts.push_back(word);
        }
        word.clear();
    };
    for (const char raw_character : caption) {
        const unsigned char character = static_cast<unsigned char>(raw_character);
        if (std::isalnum(character) != 0 || character == static_cast<unsigned char>('_')) {
            word.push_back(static_cast<char>(std::tolower(character)));
        } else {
            flush();
        }
    }
    flush();
    return concepts;
}

void VisualPatchFabric::rebuild_concept_indices() {
    mode_concept_indices_.clear();
    mode_concept_indices_.resize(modes_.size());
    for (std::size_t mode_index = 0U; mode_index < modes_.size(); ++mode_index) {
        auto& index = mode_concept_indices_[mode_index];
        index.reserve(modes_[mode_index].concepts.size());
        for (std::size_t concept_index = 0U;
             concept_index < modes_[mode_index].concepts.size();
             ++concept_index) {
            index.push_back(concept_index);
            ++training_operation_stats_.concept_index_entries_built;
        }
        std::sort(
            index.begin(), index.end(),
            [this, mode_index](const std::size_t left, const std::size_t right) {
                const std::string& left_name =
                    modes_[mode_index].concepts[left].concept_name;
                const std::string& right_name =
                    modes_[mode_index].concepts[right].concept_name;
                return left_name != right_name ? left_name < right_name : left < right;
            }
        );
    }
    ++training_operation_stats_.concept_index_rebuilds;
}

void VisualPatchFabric::rebuild_example_caption_index() {
    example_caption_index_.clear();
    example_caption_index_.reserve(examples_.size());
    for (std::size_t index = 0U; index < examples_.size(); ++index) {
        example_caption_index_[stable_string_hash(examples_[index].caption)]
            .push_back(index);
        ++training_operation_stats_.example_index_entries_built;
    }
    ++training_operation_stats_.example_index_rebuilds;
}

void VisualPatchFabric::rebuild_mode_id_index() {
    mode_id_index_.clear();
    mode_id_index_.reserve(modes_.size());
    for (std::size_t index = 0U; index < modes_.size(); ++index) {
        mode_id_index_.emplace(modes_[index].id, index);
        ++training_operation_stats_.mode_id_index_entries_rebuilt;
    }
    mode_id_index_mode_count_ = modes_.size();
    ++training_operation_stats_.mode_id_index_full_rebuilds;
}

void VisualPatchFabric::train(const ImageData& image, const std::string_view caption) {
    const PreparedImage prepared = prepare_image(image);
    std::vector<PatchRecord> patches = extract_patches(prepared.image);
    train_prepared(prepared, patches, caption);
}

VisionAnalysis VisualPatchFabric::train_and_analyze(
    const ImageData& image,
    const std::string_view caption
) {
    const PreparedImage prepared = prepare_image(image);
    std::vector<PatchRecord> patches = extract_patches(prepared.image);
    train_prepared(prepared, patches, caption);
    return analyze_prepared(image, prepared, std::move(patches));
}

void VisualPatchFabric::train_prepared(
    const PreparedImage& prepared,
    std::vector<PatchRecord>& patches,
    const std::string_view caption
) {
    if (patches.size() > config_.maximum_patches) {
        throw std::runtime_error("prepared image exceeds visual patch budget");
    }
    const std::vector<std::string> concepts = caption_concepts(caption);
    if (indexed_concept_updates_ && mode_concept_indices_.size() != modes_.size()) {
        rebuild_concept_indices();
    }
    if (indexed_example_duplicate_lookup_ && !examples_.empty() &&
        example_caption_index_.empty()) {
        rebuild_example_caption_index();
    }
    std::unordered_map<std::uint64_t, std::size_t> rebuilt_mode_id_index;
    std::unordered_map<std::uint64_t, std::size_t>* mode_indices =
        &mode_id_index_;
    if (persistent_mode_id_index_) {
        if (mode_id_index_mode_count_ != modes_.size()) {
            rebuild_mode_id_index();
        }
        mode_id_index_.reserve(modes_.size() + patches.size());
    } else {
        rebuilt_mode_id_index.reserve(modes_.size() + patches.size());
        for (std::size_t index = 0U; index < modes_.size(); ++index) {
            rebuilt_mode_id_index.emplace(modes_[index].id, index);
            ++training_operation_stats_.mode_id_index_entries_rebuilt;
        }
        ++training_operation_stats_.mode_id_index_full_rebuilds;
        mode_indices = &rebuilt_mode_id_index;
    }

    for (std::size_t begin = 0U;
         begin < patches.size();
         begin += config_.training_patch_batch) {
        const std::size_t count = std::min(
            config_.training_patch_batch,
            patches.size() - begin
        );
        std::span<PatchRecord> batch(patches.data() + begin, count);
        if (!modes_.empty()) {
            assign_existing_modes(batch);
        }
        const std::size_t modes_before_updates = modes_.size();
        const bool can_incrementally_update_router =
            incremental_sparse_router_updates_ &&
            modes_before_updates >= config_.sparse_routing_minimum_modes &&
            mode_router_revision_ == mode_revision_ &&
            mode_router_.vector_count() == modes_before_updates;
        std::vector<std::size_t> touched_mode_indices;
        touched_mode_indices.reserve(count);

        for (PatchRecord& patch : batch) {
            std::size_t mode_index = 0U;
            double similarity = patch.mode_similarity;
            if (patch.mode_id != 0U) {
                ++training_operation_stats_.mode_id_lookups;
                const auto found = mode_indices->find(patch.mode_id);
                if (found != mode_indices->end()) {
                    mode_index = found->second;
                } else {
                    patch.mode_id = 0U;
                }
            }
            if (patch.mode_id == 0U && !modes_.empty()) {
                const auto nearest = nearest_mode(patch.descriptor);
                mode_index = nearest.first;
                similarity = nearest.second;
            }

            if (modes_.empty() ||
                (similarity < config_.mode_creation_similarity &&
                 modes_.size() < config_.maximum_modes)) {
                modes_.push_back(VisualMode{
                    next_mode_id_++, patch.descriptor, 1U, {},
                });
                if (indexed_concept_updates_) {
                    mode_concept_indices_.emplace_back();
                }
                mode_index = modes_.size() - 1U;
                mode_indices->emplace(modes_[mode_index].id, mode_index);
                ++training_operation_stats_.mode_id_index_incremental_inserts;
                if (persistent_mode_id_index_) {
                    mode_id_index_mode_count_ = modes_.size();
                }
                similarity = 1.0;
            } else {
                VisualMode& mode = modes_[mode_index];
                const double adaptive_rate = std::min(
                    config_.local_learning_rate,
                    1.0 / static_cast<double>(mode.support + 1U)
                );
                backend_->local_average_update(
                    mode.prototype,
                    patch.descriptor,
                    static_cast<float>(adaptive_rate)
                );
                ++mode.support;
            }
            patch.mode_id = modes_[mode_index].id;
            patch.mode_similarity = similarity;
            touched_mode_indices.push_back(mode_index);
            VisualMode& mode = modes_[mode_index];
            for (const std::string& concept_name : concepts) {
                ++training_operation_stats_.concept_update_lookups;
                if (indexed_concept_updates_) {
                    ++training_operation_stats_.indexed_concept_lookups;
                    auto& concept_index = mode_concept_indices_[mode_index];
                    const auto found = std::lower_bound(
                        concept_index.begin(), concept_index.end(), concept_name,
                        [&mode](const std::size_t index, const std::string& name) {
                            return mode.concepts[index].concept_name < name;
                        }
                    );
                    if (found == concept_index.end()) {
                        if (mode.concepts.size() < config_.maximum_concepts_per_mode) {
                            const std::size_t index = mode.concepts.size();
                            mode.concepts.push_back(VisualConceptCount{concept_name, 1U});
                            concept_index.insert(found, index);
                        }
                    } else if (mode.concepts[*found].concept_name != concept_name) {
                        if (mode.concepts.size() < config_.maximum_concepts_per_mode) {
                            const std::size_t index = mode.concepts.size();
                            mode.concepts.push_back(VisualConceptCount{concept_name, 1U});
                            concept_index.insert(found, index);
                        }
                    } else {
                        ++mode.concepts[*found].count;
                    }
                } else {
                    auto found = mode.concepts.end();
                    for (auto candidate = mode.concepts.begin();
                         candidate != mode.concepts.end();
                         ++candidate) {
                        ++training_operation_stats_.linear_concept_comparisons;
                        if (candidate->concept_name == concept_name) {
                            found = candidate;
                            break;
                        }
                    }
                    if (found == mode.concepts.end()) {
                        if (mode.concepts.size() < config_.maximum_concepts_per_mode) {
                            mode.concepts.push_back(VisualConceptCount{concept_name, 1U});
                        }
                    } else {
                        ++found->count;
                    }
                }
            }
        }
        ++mode_revision_;
        if (mode_revision_ == 0U) {
            mode_revision_ = 1U;
        }
        if (can_incrementally_update_router) {
            std::sort(touched_mode_indices.begin(), touched_mode_indices.end());
            touched_mode_indices.erase(
                std::unique(
                    touched_mode_indices.begin(), touched_mode_indices.end()
                ),
                touched_mode_indices.end()
            );
            for (const std::size_t index : touched_mode_indices) {
                if (index < modes_before_updates) {
                    mode_router_.update(index, modes_[index].prototype);
                } else {
                    if (index != mode_router_.vector_count()) {
                        throw std::logic_error(
                            "visual sparse router append order is inconsistent"
                        );
                    }
                    mode_router_.append(modes_[index].prototype);
                }
            }
            mode_router_revision_ = mode_revision_;
        }
    }

    const std::vector<float> descriptor = global_descriptor(prepared.image, patches);
    ++training_operation_stats_.example_duplicate_lookups;
    std::size_t duplicate_index = examples_.size();
    if (indexed_example_duplicate_lookup_) {
        const auto bucket = example_caption_index_.find(stable_string_hash(caption));
        if (bucket != example_caption_index_.end()) {
            for (const std::size_t index : bucket->second) {
                ++training_operation_stats_.indexed_example_candidates;
                const VisualExample& example = examples_[index];
                if (example.caption == caption &&
                    descriptor_similarity(
                        example.global_descriptor, descriptor
                    ) > 0.9999) {
                    duplicate_index = index;
                    break;
                }
            }
        }
    } else {
        for (std::size_t index = 0U; index < examples_.size(); ++index) {
            ++training_operation_stats_.linear_example_comparisons;
            const VisualExample& example = examples_[index];
            if (example.caption == caption &&
                descriptor_similarity(example.global_descriptor, descriptor) > 0.9999) {
                duplicate_index = index;
                break;
            }
        }
    }
    if (duplicate_index != examples_.size()) {
        ++examples_[duplicate_index].support;
    } else if (examples_.size() < config_.maximum_examples) {
        const std::size_t index = examples_.size();
        examples_.push_back(VisualExample{
            next_example_id_++, descriptor, std::string(caption), concepts, 1U,
        });
        if (indexed_example_duplicate_lookup_) {
            example_caption_index_[stable_string_hash(caption)].push_back(index);
        }
    }
    ++images_seen_;
}

void VisualPatchFabric::train_file(
    const std::filesystem::path& path,
    const std::string_view caption,
    const ImageLimits limits
) {
    train(load_image(path, limits), caption);
}

std::vector<VisualRegion> VisualPatchFabric::build_regions(
    const ImageData& image,
    std::vector<PatchRecord> patches
) const {
    static_cast<void>(image);
    if (patches.empty()) {
        return {};
    }

    struct GridKey final {
        std::size_t scale{};
        std::size_t x{};
        std::size_t y{};
        [[nodiscard]] bool operator==(const GridKey&) const noexcept = default;
    };
    struct GridKeyHash final {
        [[nodiscard]] std::size_t operator()(const GridKey& key) const noexcept {
            std::size_t value = key.scale + 0x9E3779B9U;
            value ^= key.x + 0x9E3779B9U + (value << 6U) + (value >> 2U);
            value ^= key.y + 0x9E3779B9U + (value << 6U) + (value >> 2U);
            return value;
        }
    };

    std::unordered_map<GridKey, std::size_t, GridKeyHash> grid_index;
    grid_index.reserve(patches.size());
    for (std::size_t index = 0U; index < patches.size(); ++index) {
        grid_index.emplace(
            GridKey{patches[index].scale_index, patches[index].grid_x, patches[index].grid_y},
            index
        );
    }

    std::vector<bool> visited(patches.size(), false);
    std::vector<VisualRegion> regions;
    for (std::size_t start = 0U; start < patches.size(); ++start) {
        if (visited[start]) {
            continue;
        }
        visited[start] = true;
        std::queue<std::size_t> pending;
        pending.push(start);
        const std::uint64_t mode_id = patches[start].mode_id;
        const std::size_t scale_index = patches[start].scale_index;
        std::size_t minimum_x = patches[start].pixel_x;
        std::size_t minimum_y = patches[start].pixel_y;
        std::size_t maximum_x = minimum_x + patches[start].pixel_width;
        std::size_t maximum_y = minimum_y + patches[start].pixel_height;
        std::size_t count = 0U;
        double confidence = 0.0;
        while (!pending.empty()) {
            const std::size_t index = pending.front();
            pending.pop();
            const PatchRecord& patch = patches[index];
            ++count;
            confidence += patch.mode_similarity;
            minimum_x = std::min(minimum_x, patch.pixel_x);
            minimum_y = std::min(minimum_y, patch.pixel_y);
            maximum_x = std::max(maximum_x, patch.pixel_x + patch.pixel_width);
            maximum_y = std::max(maximum_y, patch.pixel_y + patch.pixel_height);
            const std::array<std::pair<int, int>, 4U> offsets{{
                {-1, 0}, {1, 0}, {0, -1}, {0, 1},
            }};
            for (const auto& [offset_x, offset_y] : offsets) {
                const auto next_x = static_cast<std::int64_t>(patch.grid_x) + offset_x;
                const auto next_y = static_cast<std::int64_t>(patch.grid_y) + offset_y;
                if (next_x < 0 || next_y < 0) {
                    continue;
                }
                const auto found = grid_index.find(GridKey{
                    scale_index,
                    static_cast<std::size_t>(next_x),
                    static_cast<std::size_t>(next_y),
                });
                if (found == grid_index.end()) {
                    continue;
                }
                const std::size_t next = found->second;
                if (!visited[next] && patches[next].mode_id == mode_id) {
                    visited[next] = true;
                    pending.push(next);
                }
            }
        }
        std::string concept_name;
        ++training_operation_stats_.region_mode_id_lookups;
        const VisualMode* mode = nullptr;
        if (persistent_mode_id_index_) {
            ++training_operation_stats_.indexed_region_mode_lookups;
            const auto found = mode_id_index_.find(mode_id);
            if (found != mode_id_index_.end()) {
                mode = &modes_[found->second];
            }
        } else {
            for (const VisualMode& candidate : modes_) {
                ++training_operation_stats_.linear_region_mode_comparisons;
                if (candidate.id == mode_id) {
                    mode = &candidate;
                    break;
                }
            }
        }
        if (mode != nullptr) {
            concept_name = top_concept(*mode);
        }
        regions.push_back(VisualRegion{
            mode_id,
            minimum_x,
            minimum_y,
            maximum_x - minimum_x,
            maximum_y - minimum_y,
            count,
            std::move(concept_name),
            count == 0U ? 0.0 : confidence / static_cast<double>(count),
        });
    }
    std::sort(
        regions.begin(), regions.end(),
        [](const VisualRegion& left, const VisualRegion& right) {
            const double left_score = static_cast<double>(left.patch_count) * left.confidence;
            const double right_score = static_cast<double>(right.patch_count) * right.confidence;
            if (left_score != right_score) {
                return left_score > right_score;
            }
            return left.mode_id < right.mode_id;
        }
    );
    if (regions.size() > config_.maximum_regions) {
        regions.resize(config_.maximum_regions);
    }
    return regions;
}

std::string VisualPatchFabric::fallback_description(
    const std::span<const float> descriptor,
    const std::size_t region_count
) const {
    if (descriptor.size() < legacy_descriptor_size) {
        return "An image was provided, but its visual descriptor is incomplete.";
    }
    const float red = descriptor[0U];
    const float green = descriptor[1U];
    const float blue = descriptor[2U];
    const float luminance = descriptor[6U];
    const float edges = descriptor[9U];
    std::string brightness = "mid-tone";
    if (luminance < 0.30F) {
        brightness = "dark";
    } else if (luminance > 0.72F) {
        brightness = "bright";
    }
    std::string color = "balanced-color";
    if (red > green + 0.08F && red > blue + 0.08F) {
        color = "red-dominant";
    } else if (green > red + 0.08F && green > blue + 0.08F) {
        color = "green-dominant";
    } else if (blue > red + 0.08F && blue > green + 0.08F) {
        color = "blue-dominant";
    }
    std::string texture = edges > 0.18F ? "high-detail" : "low-detail";
    std::ostringstream output;
    output << "A " << brightness << ", " << color << ", " << texture
           << " image with " << region_count << " visual region";
    if (region_count != 1U) {
        output << 's';
    }
    output << '.';
    return output.str();
}

VisionAnalysis VisualPatchFabric::analyze(const ImageData& image) const {
    const PreparedImage prepared = prepare_image(image);
    std::vector<PatchRecord> patches = extract_patches(prepared.image);
    return analyze_prepared(image, prepared, std::move(patches));
}

VisionAnalysis VisualPatchFabric::analyze_prepared(
    const ImageData& image,
    const PreparedImage& prepared,
    std::vector<PatchRecord> patches
) const {
    if (patches.size() > config_.maximum_patches) {
        throw std::runtime_error("prepared image exceeds visual patch budget");
    }
    assign_existing_modes(patches);
    double patch_confidence = 0.0;
    for (const PatchRecord& patch : patches) {
        patch_confidence += patch.mode_similarity;
    }
    const std::vector<float> descriptor = global_descriptor(prepared.image, patches);
    std::vector<VisualRegion> regions = build_regions(prepared.image, patches);
    for (VisualRegion& region : regions) {
        region.x = std::min(
            image.width,
            static_cast<std::size_t>(
                std::floor(static_cast<double>(region.x) * prepared.scale_x + 0.5)
            )
        );
        region.y = std::min(
            image.height,
            static_cast<std::size_t>(
                std::floor(static_cast<double>(region.y) * prepared.scale_y + 0.5)
            )
        );
        region.width = std::min(
            image.width - region.x,
            std::max<std::size_t>(
                1U,
                static_cast<std::size_t>(
                    std::floor(static_cast<double>(region.width) * prepared.scale_x + 0.5)
                )
            )
        );
        region.height = std::min(
            image.height - region.y,
            std::max<std::size_t>(
                1U,
                static_cast<std::size_t>(
                    std::floor(static_cast<double>(region.height) * prepared.scale_y + 0.5)
                )
            )
        );
    }

    const VisualExample* nearest_example = nullptr;
    double example_similarity = 0.0;
    for (const VisualExample& example : examples_) {
        const double similarity = descriptor_similarity(
            descriptor, example.global_descriptor
        );
        if (similarity > example_similarity ||
            (similarity == example_similarity && nearest_example != nullptr &&
             example.id < nearest_example->id)) {
            nearest_example = &example;
            example_similarity = similarity;
        }
    }

    std::unordered_map<std::string, double> concept_scores;
    for (const VisualRegion& region : regions) {
        if (!region.concept_name.empty()) {
            concept_scores[region.concept_name] +=
                region.confidence * static_cast<double>(region.patch_count);
        }
    }
    if (nearest_example != nullptr &&
        example_similarity >= config_.example_match_similarity) {
        for (const std::string& concept_name : nearest_example->concepts) {
            concept_scores[concept_name] += 4.0 * example_similarity;
        }
    }
    std::vector<std::pair<std::string, double>> ordered_concepts(
        concept_scores.begin(), concept_scores.end()
    );
    std::sort(
        ordered_concepts.begin(), ordered_concepts.end(),
        [](const auto& left, const auto& right) {
            if (left.second != right.second) {
                return left.second > right.second;
            }
            return left.first < right.first;
        }
    );

    VisionAnalysis analysis;
    analysis.width = image.width;
    analysis.height = image.height;
    analysis.regions = std::move(regions);
    for (std::size_t index = 0U;
         index < std::min<std::size_t>(ordered_concepts.size(), 12U);
         ++index) {
        analysis.concepts.push_back(ordered_concepts[index].first);
    }
    if (nearest_example != nullptr &&
        example_similarity >= config_.example_match_similarity) {
        analysis.description = nearest_example->caption;
        analysis.nearest_example_id = nearest_example->id;
    } else {
        analysis.description = fallback_description(descriptor, analysis.regions.size());
    }
    const double mode_confidence = patches.empty()
        ? 0.0
        : patch_confidence / static_cast<double>(patches.size());
    analysis.confidence = std::clamp(
        0.55 * mode_confidence + 0.45 * example_similarity,
        0.0,
        1.0
    );
    return analysis;
}

VisionAnalysis VisualPatchFabric::analyze_file(
    const std::filesystem::path& path,
    const ImageLimits limits
) const {
    return analyze(load_image(path, limits));
}

std::string VisualPatchFabric::grounding_text(const VisionAnalysis& analysis) const {
    std::ostringstream output;
    output << analysis.description;
    if (!analysis.concepts.empty()) {
        output << " Concepts:";
        for (const std::string& concept_name : analysis.concepts) {
            output << ' ' << concept_name;
        }
        output << '.';
    }
    if (!analysis.regions.empty()) {
        output << " Regions:";
        for (std::size_t index = 0U;
             index < std::min<std::size_t>(analysis.regions.size(), 8U);
             ++index) {
            const VisualRegion& region = analysis.regions[index];
            output << " [";
            if (!region.concept_name.empty()) {
                output << region.concept_name << ' ';
            }
            output << region.x << ',' << region.y << ' '
                   << region.width << 'x' << region.height << ']';
        }
    }
    return output.str();
}


void VisualPatchFabric::set_backend(
    const rlf::frontier::FrontierBackendKind kind
) {
    std::unique_ptr<rlf::frontier::FrontierComputeBackend> backend =
        rlf::frontier::make_frontier_backend(kind);
    if (!backend->capabilities().available) {
        throw std::runtime_error(
            "requested Solstice backend is unavailable: " +
            std::string(backend->name())
        );
    }
    backend_ = std::move(backend);
    if (!sparse_rerank_policy_explicit_) {
        batched_sparse_reranking_ = kind == rlf::frontier::FrontierBackendKind::cuda;
    }
}

rlf::frontier::FrontierBackendKind VisualPatchFabric::backend_kind() const noexcept {
    return backend_->kind();
}

rlf::frontier::BackendCapabilities VisualPatchFabric::backend_capabilities() const noexcept {
    return backend_->capabilities();
}

rlf::frontier::BackendOperationStats VisualPatchFabric::backend_operation_stats() const noexcept {
    return backend_->operation_stats();
}

SparseRouterOperationStats VisualPatchFabric::sparse_router_operation_stats() const noexcept {
    return mode_router_.operation_stats();
}

VisualTrainingOperationStats VisualPatchFabric::training_operation_stats() const noexcept {
    return training_operation_stats_;
}

void VisualPatchFabric::set_incremental_sparse_router_updates(
    const bool enabled
) noexcept {
    incremental_sparse_router_updates_ = enabled;
}

void VisualPatchFabric::set_batched_sparse_reranking(const bool enabled) noexcept {
    batched_sparse_reranking_ = enabled;
    sparse_rerank_policy_explicit_ = true;
}

void VisualPatchFabric::set_indexed_concept_updates(const bool enabled) {
    indexed_concept_updates_ = enabled;
    if (enabled) {
        rebuild_concept_indices();
    } else {
        std::vector<std::vector<std::size_t>> empty;
        mode_concept_indices_.swap(empty);
    }
}

void VisualPatchFabric::set_indexed_example_duplicate_lookup(
    const bool enabled
) {
    indexed_example_duplicate_lookup_ = enabled;
    if (enabled) {
        rebuild_example_caption_index();
    } else {
        decltype(example_caption_index_) empty;
        example_caption_index_.swap(empty);
    }
}

void VisualPatchFabric::set_persistent_mode_id_index(const bool enabled) {
    persistent_mode_id_index_ = enabled;
    if (enabled) {
        rebuild_mode_id_index();
    } else {
        decltype(mode_id_index_) empty;
        mode_id_index_.swap(empty);
        mode_id_index_mode_count_ = 0U;
    }
}

const VisionConfig& VisualPatchFabric::config() const noexcept { return config_; }
std::span<const VisualMode> VisualPatchFabric::modes() const noexcept { return modes_; }
std::span<const VisualExample> VisualPatchFabric::examples() const noexcept { return examples_; }
std::uint64_t VisualPatchFabric::images_seen() const noexcept { return images_seen_; }

VisionSnapshot VisualPatchFabric::snapshot() const {
    return VisionSnapshot{
        config_, next_mode_id_, next_example_id_, images_seen_, modes_, examples_,
    };
}

VisualPatchFabric VisualPatchFabric::from_snapshot(VisionSnapshot snapshot) {
    VisualPatchFabric fabric(snapshot.config);
    if (snapshot.modes.size() > snapshot.config.maximum_modes ||
        snapshot.examples.size() > snapshot.config.maximum_examples ||
        snapshot.next_mode_id == 0U || snapshot.next_example_id == 0U) {
        throw std::invalid_argument("invalid Solstice vision snapshot dimensions");
    }
    for (const VisualMode& mode : snapshot.modes) {
        if (mode.id == 0U || mode.prototype.size() != snapshot.config.descriptor_dimensions ||
            mode.concepts.size() > snapshot.config.maximum_concepts_per_mode) {
            throw std::invalid_argument("invalid Solstice visual mode snapshot");
        }
    }
    for (const VisualExample& example : snapshot.examples) {
        if (example.id == 0U || example.global_descriptor.size() != snapshot.config.descriptor_dimensions) {
            throw std::invalid_argument("invalid Solstice visual example snapshot");
        }
    }
    fabric.next_mode_id_ = snapshot.next_mode_id;
    fabric.next_example_id_ = snapshot.next_example_id;
    fabric.images_seen_ = snapshot.images_seen;
    fabric.mode_revision_ = 1U;
    fabric.modes_ = std::move(snapshot.modes);
    fabric.examples_ = std::move(snapshot.examples);
    if (fabric.indexed_concept_updates_) {
        fabric.rebuild_concept_indices();
    }
    if (fabric.indexed_example_duplicate_lookup_) {
        fabric.rebuild_example_caption_index();
    }
    if (fabric.persistent_mode_id_index_) {
        fabric.rebuild_mode_id_index();
    }
    return fabric;
}

std::uint64_t VisualPatchFabric::deterministic_hash() const noexcept {
    std::uint64_t hash = fnv_offset_basis;
    hash_u64(hash, images_seen_);
    for (const VisualMode& mode : modes_) {
        hash_u64(hash, mode.id);
        hash_u64(hash, mode.support);
        for (const float value : mode.prototype) {
            hash_float(hash, value);
        }
        std::vector<VisualConceptCount> concepts = mode.concepts;
        std::sort(
            concepts.begin(), concepts.end(),
            [](const VisualConceptCount& left, const VisualConceptCount& right) {
                return left.concept_name < right.concept_name;
            }
        );
        for (const VisualConceptCount& concept_name : concepts) {
            hash_string(hash, concept_name.concept_name);
            hash_u64(hash, concept_name.count);
        }
    }
    for (const VisualExample& example : examples_) {
        hash_u64(hash, example.id);
        hash_u64(hash, example.support);
        hash_string(hash, example.caption);
        for (const float value : example.global_descriptor) {
            hash_float(hash, value);
        }
    }
    return hash;
}

}  // namespace rlf::solstice
