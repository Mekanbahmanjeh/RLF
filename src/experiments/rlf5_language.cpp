#include "rlf/experiments/rlf5_language.hpp"

#include "rlf/core/deterministic_rng.hpp"
#include "rlf/storage/rlf5_checkpoint.hpp"

#include <algorithm>
#include <array>
#include <bit>
#include <chrono>
#include <cctype>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <iomanip>
#include <limits>
#include <map>
#include <set>
#include <span>
#include <sstream>
#include <stdexcept>
#include <string>
#include <string_view>
#include <unordered_map>
#include <unordered_set>
#include <utility>
#include <vector>

namespace rlf::experiments {
namespace {

constexpr std::uint64_t fnv_offset_basis = 14'695'981'039'346'656'037ULL;
constexpr std::uint64_t fnv_prime = 1'099'511'628'211ULL;
constexpr double probability_floor = 1.0e-12;
constexpr double ln2 = 0.693147180559945309417232121458176568;

void hash_u64(std::uint64_t& hash, const std::uint64_t value) noexcept {
    for (unsigned int byte = 0U; byte < 8U; ++byte) {
        hash ^= (value >> (byte * 8U)) & 0xFFULL;
        hash *= fnv_prime;
    }
}

void hash_double(std::uint64_t& hash, const double value) noexcept {
    hash_u64(hash, std::bit_cast<std::uint64_t>(value));
}

void hash_string(std::uint64_t& hash, const std::string_view value) noexcept {
    hash_u64(hash, value.size());
    for (const char raw : value) {
        hash ^= static_cast<unsigned char>(raw);
        hash *= fnv_prime;
    }
}

[[nodiscard]] std::string format_hash(const std::uint64_t value) {
    std::ostringstream output;
    output << std::hex << std::setw(16) << std::setfill('0') << value;
    return output.str();
}

[[nodiscard]] std::string json_escape(const std::string_view value) {
    std::string result;
    result.reserve(value.size() + 8U);
    for (const char character : value) {
        switch (character) {
            case '\\': result += "\\\\"; break;
            case '"': result += "\\\""; break;
            case '\n': result += "\\n"; break;
            case '\r': result += "\\r"; break;
            case '\t': result += "\\t"; break;
            default:
                if (static_cast<unsigned char>(character) < 0x20U) {
                    result += '?';
                } else {
                    result.push_back(character);
                }
                break;
        }
    }
    return result;
}

void validate_config(const Rlf5Config& config) {
    if (config.phase_dimension < 8U || config.raw_training_sentences < 256U ||
        config.supervised_training_examples < 256U ||
        config.evaluation_examples < 64U || config.qa_episodes < 32U ||
        config.free_generation_samples < 8U || config.maximum_lexemes < 320U ||
        config.maximum_merges == 0U || config.minimum_pair_support < 2U ||
        config.maximum_context_order == 0U ||
        config.minimum_context_support == 0U ||
        config.maximum_constructions == 0U ||
        config.minimum_construction_support == 0U ||
        config.maximum_generation_tokens < 16U || config.holdout_modulus < 3U) {
        throw std::invalid_argument("invalid RLF-5 experiment configuration");
    }
}

struct PredicateDefinition final {
    std::string value;
    std::vector<std::string> present;
    std::vector<std::string> base;
    std::vector<std::string> past;
};

struct RenderedExample final {
    std::string text;
    core::LanguageFrame frame;
};

[[nodiscard]] std::string frame_key(const core::LanguageFrame& frame) {
    std::string key;
    key.reserve(128U);
    key += std::to_string(static_cast<unsigned int>(frame.act));
    const std::array<std::string_view, 6U> values{
        frame.predicate, frame.agent, frame.patient, frame.agent_attribute,
        frame.patient_attribute, frame.location,
    };
    for (const auto value : values) {
        key.push_back('|');
        key += value;
    }
    return key;
}

[[nodiscard]] std::vector<std::string> split_words(const std::string_view text) {
    std::vector<std::string> result;
    std::string current;
    for (const char raw : text) {
        const auto character = static_cast<unsigned char>(raw);
        if (std::isalnum(character) != 0 || character == '\'' || character == '-') {
            current.push_back(static_cast<char>(std::tolower(character)));
        } else if (!current.empty()) {
            result.push_back(std::move(current));
            current.clear();
        }
    }
    if (!current.empty()) {
        result.push_back(std::move(current));
    }
    return result;
}

class CompositionalLanguageWorld final {
public:
    explicit CompositionalLanguageWorld(
        const Rlf5Config& config,
        const std::uint64_t seed
    ) : config_(config), seed_(seed) {}

    [[nodiscard]] std::string raw_corpus(const std::size_t sentences) const {
        core::DeterministicRng rng(seed_ ^ 0x524157434F525055ULL);
        std::string corpus;
        corpus.reserve(sentences * 72U);
        for (std::size_t index = 0U; index < sentences; ++index) {
            const auto frame = sample_frame(rng, false);
            const std::size_t choice = rng.uniform_index(10U);
            if (choice < 6U) {
                corpus += render_statement(frame, rng);
            } else if (choice == 6U) {
                corpus += render_question(frame, core::LanguageAct::query_patient, rng).text;
            } else if (choice == 7U) {
                corpus += render_question(frame, core::LanguageAct::query_agent, rng).text;
            } else if (choice == 8U) {
                corpus += render_question(frame, core::LanguageAct::query_location, rng).text;
            } else {
                const auto answer = answer_frame(
                    frame,
                    static_cast<core::LanguageAct>(
                        static_cast<unsigned int>(core::LanguageAct::answer_agent) +
                        rng.uniform_index(3U)
                    )
                );
                corpus += render_answer(answer, rng);
            }
        }
        return corpus;
    }

    [[nodiscard]] std::vector<core::LanguageSupervisedExample> supervised(
        const std::size_t count
    ) const {
        core::DeterministicRng rng(seed_ ^ 0x5355504552564953ULL);
        std::vector<core::LanguageSupervisedExample> result;
        result.reserve(count);
        for (std::size_t index = 0U; index < count; ++index) {
            const auto frame = sample_frame(rng, false);
            const std::size_t act_selector = index % 7U;
            RenderedExample rendered;
            if (act_selector == 0U || act_selector == 1U) {
                rendered = {render_statement(frame, rng), frame};
            } else if (act_selector == 2U) {
                rendered = render_question(frame, core::LanguageAct::query_patient, rng);
            } else if (act_selector == 3U) {
                rendered = render_question(frame, core::LanguageAct::query_agent, rng);
            } else if (act_selector == 4U) {
                rendered = render_question(frame, core::LanguageAct::query_location, rng);
            } else if (act_selector == 5U) {
                const auto answer = answer_frame(frame, core::LanguageAct::answer_patient);
                rendered = {render_answer(answer, rng), answer};
            } else {
                const auto act = rng.uniform_index(2U) == 0U
                    ? core::LanguageAct::answer_agent
                    : core::LanguageAct::answer_location;
                const auto answer = answer_frame(frame, act);
                rendered = {render_answer(answer, rng), answer};
            }
            result.push_back({std::move(rendered.text), std::move(rendered.frame)});
        }
        return result;
    }

    [[nodiscard]] std::vector<RenderedExample> evaluation_statements(
        const std::size_t count,
        const std::uint64_t salt = 0ULL
    ) const {
        core::DeterministicRng rng(seed_ ^ 0x4556414C53544154ULL ^ salt);
        std::vector<RenderedExample> result;
        result.reserve(count);
        for (std::size_t index = 0U; index < count; ++index) {
            const auto frame = sample_frame(rng, true);
            result.push_back({render_statement(frame, rng), frame});
        }
        return result;
    }

    struct QaEpisode final {
        std::vector<std::string> context;
        std::vector<core::LanguageFrame> facts;
        std::string question;
        core::LanguageFrame query;
        core::LanguageFrame answer;
        std::size_t target_index{};
    };

    [[nodiscard]] QaEpisode qa_episode(
        core::DeterministicRng& rng
    ) const {
        QaEpisode episode;
        episode.context.reserve(4U);
        episode.facts.reserve(4U);
        const auto target = sample_frame(rng, true);
        const auto query_act = static_cast<core::LanguageAct>(
            static_cast<unsigned int>(core::LanguageAct::query_agent) +
            rng.uniform_index(3U)
        );
        episode.target_index = rng.uniform_index(4U);
        for (std::size_t index = 0U; index < 4U; ++index) {
            core::LanguageFrame fact = index == episode.target_index
                ? target
                : sample_distractor(rng, target, query_act);
            episode.context.push_back(render_statement(fact, rng));
            episode.facts.push_back(std::move(fact));
        }
        const auto rendered = render_question(target, query_act, rng);
        episode.question = rendered.text;
        episode.query = rendered.frame;
        const auto answer_act = query_act == core::LanguageAct::query_agent
            ? core::LanguageAct::answer_agent
            : query_act == core::LanguageAct::query_patient
                ? core::LanguageAct::answer_patient
                : core::LanguageAct::answer_location;
        episode.answer = answer_frame(target, answer_act);
        return episode;
    }

    [[nodiscard]] std::uint64_t corpus_hash(const std::string_view corpus) const {
        std::uint64_t hash = fnv_offset_basis;
        hash_string(hash, corpus);
        return hash;
    }

private:
    [[nodiscard]] bool is_holdout(
        const std::size_t agent,
        const std::size_t patient,
        const std::size_t agent_attribute,
        const std::size_t patient_attribute,
        const std::size_t location,
        const std::size_t predicate
    ) const noexcept {
        std::uint64_t code = agent;
        code = code * patients_.size() + patient;
        code = code * attributes_.size() + agent_attribute;
        code = code * attributes_.size() + patient_attribute;
        code = code * locations_.size() + location;
        code = code * predicates_.size() + predicate;
        return code % config_.holdout_modulus == 0ULL;
    }

    [[nodiscard]] core::LanguageFrame sample_frame(
        core::DeterministicRng& rng,
        const bool holdout
    ) const {
        for (std::size_t attempt = 0U; attempt < 10'000U; ++attempt) {
            const std::size_t agent = rng.uniform_index(agents_.size());
            const std::size_t patient = rng.uniform_index(patients_.size());
            const std::size_t agent_attribute = rng.uniform_index(attributes_.size());
            const std::size_t patient_attribute = rng.uniform_index(attributes_.size());
            const std::size_t location = rng.uniform_index(locations_.size());
            const std::size_t predicate = rng.uniform_index(predicates_.size());
            if (is_holdout(
                    agent, patient, agent_attribute, patient_attribute,
                    location, predicate
                ) != holdout) {
                continue;
            }
            return {
                core::LanguageAct::statement,
                predicates_[predicate].value,
                agents_[agent],
                patients_[patient],
                attributes_[agent_attribute],
                attributes_[patient_attribute],
                locations_[location],
            };
        }
        throw std::runtime_error("unable to sample RLF-5 compositional frame");
    }

    [[nodiscard]] core::LanguageFrame sample_distractor(
        core::DeterministicRng& rng,
        const core::LanguageFrame& target,
        const core::LanguageAct query_act
    ) const {
        for (std::size_t attempt = 0U; attempt < 1'000U; ++attempt) {
            auto frame = sample_frame(rng, true);
            bool differs_on_observed_role = frame.predicate != target.predicate;
            if (query_act != core::LanguageAct::query_agent) {
                differs_on_observed_role = differs_on_observed_role ||
                    frame.agent != target.agent ||
                    frame.agent_attribute != target.agent_attribute;
            }
            if (query_act != core::LanguageAct::query_patient) {
                differs_on_observed_role = differs_on_observed_role ||
                    frame.patient != target.patient ||
                    frame.patient_attribute != target.patient_attribute;
            }
            if (query_act != core::LanguageAct::query_location) {
                differs_on_observed_role = differs_on_observed_role ||
                    frame.location != target.location;
            }
            if (differs_on_observed_role) {
                return frame;
            }
        }
        throw std::runtime_error("unable to sample RLF-5 distractor");
    }

    [[nodiscard]] const PredicateDefinition& predicate(
        const std::string_view value
    ) const {
        const auto found = std::find_if(
            predicates_.begin(), predicates_.end(),
            [value](const PredicateDefinition& item) {
                return item.value == value;
            }
        );
        if (found == predicates_.end()) {
            throw std::out_of_range("unknown RLF-5 predicate");
        }
        return *found;
    }

    [[nodiscard]] std::string render_statement(
        const core::LanguageFrame& frame,
        core::DeterministicRng& rng
    ) const {
        const auto& definition = predicate(frame.predicate);
        const auto& present = definition.present[
            rng.uniform_index(definition.present.size())
        ];
        const auto& past = definition.past[
            rng.uniform_index(definition.past.size())
        ];
        switch (rng.uniform_index(5U)) {
            case 0U:
                return "the " + frame.agent_attribute + " " + frame.agent +
                    " " + present + " the " + frame.patient_attribute + " " +
                    frame.patient + " in the " + frame.location + ".\n";
            case 1U:
                return "in the " + frame.location + ", the " +
                    frame.agent_attribute + " " + frame.agent + " " + present +
                    " the " + frame.patient_attribute + " " + frame.patient + ".\n";
            case 2U:
                return "the " + frame.patient_attribute + " " + frame.patient +
                    " is " + past + " by the " + frame.agent_attribute + " " +
                    frame.agent + " in the " + frame.location + ".\n";
            case 3U:
                return "near the " + frame.location + ", the " +
                    frame.patient_attribute + " " + frame.patient + " is " +
                    past + " by the " + frame.agent_attribute + " " +
                    frame.agent + ".\n";
            default:
                return "the " + frame.agent_attribute + " " + frame.agent +
                    " " + present + " the " + frame.patient_attribute + " " +
                    frame.patient + " near the " + frame.location + ".\n";
        }
    }

    [[nodiscard]] RenderedExample render_question(
        const core::LanguageFrame& fact,
        const core::LanguageAct act,
        core::DeterministicRng& rng
    ) const {
        const auto& definition = predicate(fact.predicate);
        const auto& base = definition.base[rng.uniform_index(definition.base.size())];
        const auto& present = definition.present[
            rng.uniform_index(definition.present.size())
        ];
        const auto& past = definition.past[rng.uniform_index(definition.past.size())];
        core::LanguageFrame frame = fact;
        frame.act = act;
        std::string text;
        if (act == core::LanguageAct::query_patient) {
            frame.patient.clear();
            frame.patient_attribute.clear();
            if (rng.uniform_index(2U) == 0U) {
                text = "what does the " + fact.agent_attribute + " " + fact.agent +
                    " " + base + " in the " + fact.location + "?\n";
            } else {
                text = "which object is " + past + " by the " +
                    fact.agent_attribute + " " + fact.agent + " in the " +
                    fact.location + "?\n";
            }
        } else if (act == core::LanguageAct::query_agent) {
            frame.agent.clear();
            frame.agent_attribute.clear();
            if (rng.uniform_index(2U) == 0U) {
                text = "who " + present + " the " + fact.patient_attribute +
                    " " + fact.patient + " in the " + fact.location + "?\n";
            } else {
                text = "which creature " + present + " the " +
                    fact.patient_attribute + " " + fact.patient + " near the " +
                    fact.location + "?\n";
            }
        } else if (act == core::LanguageAct::query_location) {
            frame.location.clear();
            if (rng.uniform_index(2U) == 0U) {
                text = "where does the " + fact.agent_attribute + " " + fact.agent +
                    " " + base + " the " + fact.patient_attribute + " " +
                    fact.patient + "?\n";
            } else {
                text = "where is the " + fact.patient_attribute + " " +
                    fact.patient + " " + past + " by the " +
                    fact.agent_attribute + " " + fact.agent + "?\n";
            }
        } else {
            throw std::invalid_argument("invalid RLF-5 question act");
        }
        return {std::move(text), std::move(frame)};
    }

    [[nodiscard]] static core::LanguageFrame answer_frame(
        const core::LanguageFrame& fact,
        const core::LanguageAct act
    ) {
        core::LanguageFrame answer;
        answer.act = act;
        if (act == core::LanguageAct::answer_agent) {
            answer.agent = fact.agent;
            answer.agent_attribute = fact.agent_attribute;
        } else if (act == core::LanguageAct::answer_patient) {
            answer.patient = fact.patient;
            answer.patient_attribute = fact.patient_attribute;
        } else if (act == core::LanguageAct::answer_location) {
            answer.location = fact.location;
        }
        return answer;
    }

    [[nodiscard]] static std::string render_answer(
        const core::LanguageFrame& answer,
        core::DeterministicRng& rng
    ) {
        const bool article = rng.uniform_index(3U) != 0U;
        if (answer.act == core::LanguageAct::answer_agent) {
            return (article ? "the " : "") + answer.agent_attribute + " " +
                answer.agent + ".\n";
        }
        if (answer.act == core::LanguageAct::answer_patient) {
            return (article ? "the " : "") + answer.patient_attribute + " " +
                answer.patient + ".\n";
        }
        if (answer.act == core::LanguageAct::answer_location) {
            return (article ? "the " : "") + answer.location + ".\n";
        }
        return {};
    }

    Rlf5Config config_;
    std::uint64_t seed_{};
    const std::vector<std::string> agents_{
        "fox", "raven", "otter", "badger", "sailor", "baker",
        "doctor", "artist", "robot", "pilot", "farmer", "scholar",
    };
    const std::vector<std::string> patients_{
        "key", "lantern", "book", "compass", "apple", "violin",
        "map", "cup", "letter", "hammer", "camera", "crystal",
    };
    const std::vector<std::string> attributes_{
        "red", "blue", "golden", "silver", "quiet", "bright",
        "small", "ancient",
    };
    const std::vector<std::string> locations_{
        "garden", "tower", "harbor", "library", "workshop", "forest",
        "station", "gallery",
    };
    const std::vector<PredicateDefinition> predicates_{
        {"carry", {"carries", "transports"}, {"carry", "transport"},
         {"carried", "transported"}},
        {"place", {"places", "positions"}, {"place", "position"},
         {"placed", "positioned"}},
        {"admire", {"admires", "praises"}, {"admire", "praise"},
         {"admired", "praised"}},
        {"watch", {"watches", "observes"}, {"watch", "observe"},
         {"watched", "observed"}},
        {"locate", {"locates", "discovers"}, {"locate", "discover"},
         {"located", "discovered"}},
        {"move", {"moves", "shifts"}, {"move", "shift"},
         {"moved", "shifted"}},
    };
};

class ByteNgram final {
public:
    explicit ByteNgram(const std::size_t order) : order_(order) {}

    void train(const std::string_view corpus) {
        std::string history;
        for (const char character : corpus) {
            const std::size_t maximum = std::min(order_, history.size());
            for (std::size_t length = 0U; length <= maximum; ++length) {
                const std::string key = length == 0U
                    ? std::string{}
                    : history.substr(history.size() - length, length);
                ++counts_[key][static_cast<unsigned char>(character)];
            }
            if (character == '\n') {
                history.clear();
            } else {
                history.push_back(character);
                if (history.size() > order_) {
                    history.erase(history.begin());
                }
            }
        }
    }

    [[nodiscard]] double bits_per_byte(const std::string_view text) const {
        if (text.empty()) {
            return 0.0;
        }
        std::string history;
        double nll = 0.0;
        for (const char character : text) {
            const auto target = static_cast<unsigned char>(character);
            const auto* distribution = find(history);
            double probability = 1.0 / 256.0;
            if (distribution != nullptr) {
                double total = 0.0;
                for (const auto& [symbol, count] : *distribution) {
                    static_cast<void>(symbol);
                    total += static_cast<double>(count) + 0.1;
                }
                const auto found = distribution->find(target);
                const double count = found == distribution->end()
                    ? 0.0
                    : static_cast<double>(found->second);
                probability = (count + 0.1) /
                    (total + 0.1 * (256.0 - static_cast<double>(distribution->size())));
            }
            nll -= std::log(std::max(probability, probability_floor));
            if (character == '\n') {
                history.clear();
            } else {
                history.push_back(character);
                if (history.size() > order_) {
                    history.erase(history.begin());
                }
            }
        }
        return nll / (ln2 * static_cast<double>(text.size()));
    }

private:
    using Distribution = std::unordered_map<unsigned char, std::uint64_t>;

    [[nodiscard]] const Distribution* find(const std::string& history) const {
        const std::size_t maximum = std::min(order_, history.size());
        for (std::size_t length = maximum + 1U; length > 0U; --length) {
            const std::size_t actual = length - 1U;
            const std::string key = actual == 0U
                ? std::string{}
                : history.substr(history.size() - actual, actual);
            const auto found = counts_.find(key);
            if (found != counts_.end()) {
                return &found->second;
            }
        }
        return nullptr;
    }

    std::size_t order_{};
    std::unordered_map<std::string, Distribution> counts_;
};

struct StringVectorHash final {
    [[nodiscard]] std::size_t operator()(
        const std::vector<std::string>& value
    ) const noexcept {
        std::uint64_t hash = fnv_offset_basis;
        for (const auto& item : value) {
            hash_string(hash, item);
        }
        return static_cast<std::size_t>(hash);
    }
};

class WordNgram final {
public:
    explicit WordNgram(const std::size_t order) : order_(order) {}

    void train(const std::string_view corpus) {
        const auto words = split_words(corpus);
        std::vector<std::string> history;
        for (const auto& word : words) {
            const std::size_t maximum = std::min(order_, history.size());
            for (std::size_t length = 0U; length <= maximum; ++length) {
                std::vector<std::string> key;
                if (length > 0U) {
                    key.assign(
                        history.end() - static_cast<std::ptrdiff_t>(length),
                        history.end()
                    );
                }
                ++counts_[key][word];
            }
            vocabulary_.insert(word);
            history.push_back(word);
            if (history.size() > order_) {
                history.erase(history.begin());
            }
        }
    }

    [[nodiscard]] double bits_per_byte(const std::string_view text) const {
        const auto words = split_words(text);
        if (words.empty() || text.empty()) {
            return 0.0;
        }
        std::vector<std::string> history;
        double nll = 0.0;
        for (const auto& word : words) {
            const auto* distribution = find(history);
            double probability = 1.0 /
                static_cast<double>(std::max<std::size_t>(1U, vocabulary_.size()));
            if (distribution != nullptr) {
                double total = 0.0;
                for (const auto& [item, count] : *distribution) {
                    static_cast<void>(item);
                    total += static_cast<double>(count) + 0.1;
                }
                const auto found = distribution->find(word);
                const double count = found == distribution->end()
                    ? 0.0
                    : static_cast<double>(found->second);
                probability = (count + 0.1) /
                    (total + 0.1 * static_cast<double>(
                        std::max<std::size_t>(1U, vocabulary_.size() - distribution->size())
                    ));
            }
            nll -= std::log(std::max(probability, probability_floor));
            history.push_back(word);
            if (history.size() > order_) {
                history.erase(history.begin());
            }
        }
        return nll / (ln2 * static_cast<double>(text.size()));
    }

private:
    using Distribution = std::unordered_map<std::string, std::uint64_t>;

    [[nodiscard]] const Distribution* find(
        const std::vector<std::string>& history
    ) const {
        const std::size_t maximum = std::min(order_, history.size());
        for (std::size_t length = maximum + 1U; length > 0U; --length) {
            const std::size_t actual = length - 1U;
            std::vector<std::string> key;
            if (actual > 0U) {
                key.assign(
                    history.end() - static_cast<std::ptrdiff_t>(actual),
                    history.end()
                );
            }
            const auto found = counts_.find(key);
            if (found != counts_.end()) {
                return &found->second;
            }
        }
        return nullptr;
    }

    std::size_t order_{};
    std::unordered_map<std::vector<std::string>, Distribution, StringVectorHash> counts_;
    std::unordered_set<std::string> vocabulary_;
};

class NearestExampleBaseline final {
public:
    explicit NearestExampleBaseline(
        const std::span<const core::LanguageSupervisedExample> examples
    ) {
        entries_.reserve(examples.size());
        for (const auto& example : examples) {
            entries_.push_back({split_words(example.text), example.frame});
        }
    }

    [[nodiscard]] core::LanguageFrame predict(const std::string_view text) const {
        const auto words = split_words(text);
        std::unordered_set<std::string> query(words.begin(), words.end());
        const Entry* best = nullptr;
        double best_score = -1.0;
        for (const auto& entry : entries_) {
            std::unordered_set<std::string> candidate(
                entry.words.begin(), entry.words.end()
            );
            std::size_t intersection = 0U;
            for (const auto& word : query) {
                if (candidate.contains(word)) {
                    ++intersection;
                }
            }
            const std::size_t union_size = query.size() + candidate.size() - intersection;
            const double score = union_size > 0U
                ? static_cast<double>(intersection) /
                    static_cast<double>(union_size)
                : 0.0;
            if (score > best_score) {
                best_score = score;
                best = &entry;
            }
        }
        return best == nullptr ? core::LanguageFrame{} : best->frame;
    }

private:
    struct Entry final {
        std::vector<std::string> words;
        core::LanguageFrame frame;
    };
    std::vector<Entry> entries_;
};

class BagOfWordsBaseline final {
public:
    explicit BagOfWordsBaseline(
        const std::span<const core::LanguageSupervisedExample> examples
    ) {
        for (const auto& example : examples) {
            const auto word_list = split_words(example.text);
            const std::unordered_set<std::string> words(
                word_list.begin(), word_list.end()
            );
            for (const auto& word : words) {
                ++act_counts_[word][example.frame.act];
                add(word, core::LanguageRole::predicate, example.frame.predicate);
                add(word, core::LanguageRole::agent, example.frame.agent);
                add(word, core::LanguageRole::patient, example.frame.patient);
                add(word, core::LanguageRole::agent_attribute,
                    example.frame.agent_attribute);
                add(word, core::LanguageRole::patient_attribute,
                    example.frame.patient_attribute);
                add(word, core::LanguageRole::location, example.frame.location);
            }
        }
    }

    [[nodiscard]] core::LanguageFrame predict(const std::string_view text) const {
        const auto words = split_words(text);
        core::LanguageFrame frame;
        frame.act = best_act(words);
        frame.predicate = best_value(words, core::LanguageRole::predicate);
        frame.agent = best_value(words, core::LanguageRole::agent);
        frame.patient = best_value(words, core::LanguageRole::patient);
        frame.agent_attribute = best_value(words, core::LanguageRole::agent_attribute);
        frame.patient_attribute = best_value(words, core::LanguageRole::patient_attribute);
        frame.location = best_value(words, core::LanguageRole::location);
        if (frame.act == core::LanguageAct::query_agent) {
            frame.agent.clear();
            frame.agent_attribute.clear();
        } else if (frame.act == core::LanguageAct::query_patient) {
            frame.patient.clear();
            frame.patient_attribute.clear();
        } else if (frame.act == core::LanguageAct::query_location) {
            frame.location.clear();
        }
        return frame;
    }

private:
    struct RoleKey final {
        std::string word;
        core::LanguageRole role{core::LanguageRole::predicate};
        [[nodiscard]] bool operator==(const RoleKey&) const noexcept = default;
    };
    struct RoleKeyHash final {
        [[nodiscard]] std::size_t operator()(const RoleKey& key) const noexcept {
            std::uint64_t hash = fnv_offset_basis;
            hash_string(hash, key.word);
            hash_u64(hash, static_cast<std::uint64_t>(key.role));
            return static_cast<std::size_t>(hash);
        }
    };

    void add(
        const std::string& word,
        const core::LanguageRole role,
        const std::string& value
    ) {
        if (!value.empty()) {
            ++value_counts_[{word, role}][value];
        }
    }

    [[nodiscard]] core::LanguageAct best_act(
        const std::span<const std::string> words
    ) const {
        std::map<core::LanguageAct, std::uint64_t> scores;
        for (const auto& word : words) {
            const auto found = act_counts_.find(word);
            if (found == act_counts_.end()) {
                continue;
            }
            for (const auto& [act, count] : found->second) {
                scores[act] += count;
            }
        }
        core::LanguageAct best = core::LanguageAct::statement;
        std::uint64_t best_score = 0ULL;
        for (const auto& [act, score] : scores) {
            if (score > best_score) {
                best = act;
                best_score = score;
            }
        }
        return best;
    }

    [[nodiscard]] std::string best_value(
        const std::span<const std::string> words,
        const core::LanguageRole role
    ) const {
        std::unordered_map<std::string, std::uint64_t> scores;
        for (const auto& word : words) {
            const auto found = value_counts_.find({word, role});
            if (found == value_counts_.end()) {
                continue;
            }
            for (const auto& [value, count] : found->second) {
                scores[value] += count;
            }
        }
        std::string best;
        std::uint64_t best_score = 0ULL;
        for (const auto& [value, score] : scores) {
            if (score > best_score ||
                (score == best_score && (best.empty() || value < best))) {
                best = value;
                best_score = score;
            }
        }
        return best;
    }

    std::unordered_map<std::string,
        std::map<core::LanguageAct, std::uint64_t>> act_counts_;
    std::unordered_map<RoleKey,
        std::unordered_map<std::string, std::uint64_t>, RoleKeyHash> value_counts_;
};

[[nodiscard]] core::LanguageFabricConfig fabric_config(
    const Rlf5Config& config
) {
    core::LanguageFabricConfig result;
    result.phase_dimension = config.phase_dimension;
    result.maximum_lexemes = config.maximum_lexemes;
    result.maximum_merges = config.maximum_merges;
    result.minimum_pair_support = config.minimum_pair_support;
    result.maximum_context_order = config.maximum_context_order;
    result.minimum_context_support = config.minimum_context_support;
    result.maximum_constructions = config.maximum_constructions;
    result.minimum_construction_support = config.minimum_construction_support;
    result.maximum_generation_tokens = config.maximum_generation_tokens;
    result.maximum_semantic_values = 4'096U;
    result.maximum_surfaces_per_concept = 12U;
    result.minimum_lexical_score = 0.05;
    return result;
}

[[nodiscard]] std::string evaluation_corpus(
    const std::span<const RenderedExample> examples
) {
    std::string corpus;
    for (const auto& example : examples) {
        corpus += example.text;
    }
    return corpus;
}

[[nodiscard]] Rlf5SegmentationMetrics evaluate_segmentation(
    const core::LanguageFabric& fabric,
    const std::string_view corpus
) {
    Rlf5SegmentationMetrics result;
    result.raw_bytes = corpus.size();
    const auto tokens = fabric.encode(corpus);
    result.encoded_tokens = tokens.size();
    result.learned_lexemes = fabric.lexemes().size();
    result.learned_merges = fabric.merges().size();
    result.compression_ratio = tokens.empty()
        ? 0.0
        : static_cast<double>(corpus.size()) /
            static_cast<double>(tokens.size());
    result.exact_roundtrip_rate = fabric.decode(tokens) == corpus ? 1.0 : 0.0;

    std::unordered_set<std::size_t> predicted;
    std::size_t position = 0U;
    for (const auto token : tokens) {
        position += fabric.lexeme_by_id(token).bytes.size();
        if (position < corpus.size() && position > 0U) {
            const auto previous = static_cast<unsigned char>(corpus[position - 1U]);
            if (std::isalnum(previous) != 0 || previous == '\'' || previous == '-') {
                predicted.insert(position);
            }
        }
    }
    std::unordered_set<std::size_t> truth;
    for (std::size_t index = 1U; index <= corpus.size(); ++index) {
        const auto previous = static_cast<unsigned char>(corpus[index - 1U]);
        const bool previous_word = std::isalnum(previous) != 0 ||
            previous == '\'' || previous == '-';
        bool next_word = false;
        if (index < corpus.size()) {
            const auto next = static_cast<unsigned char>(corpus[index]);
            next_word = std::isalnum(next) != 0 || next == '\'' || next == '-';
        }
        if (previous_word && !next_word) {
            truth.insert(index);
        }
    }
    std::size_t intersection = 0U;
    for (const auto boundary : predicted) {
        if (truth.contains(boundary)) {
            ++intersection;
        }
    }
    result.boundary_precision = predicted.empty()
        ? 0.0
        : static_cast<double>(intersection) / static_cast<double>(predicted.size());
    result.boundary_recall = truth.empty()
        ? 0.0
        : static_cast<double>(intersection) / static_cast<double>(truth.size());
    result.boundary_f1 = result.boundary_precision + result.boundary_recall > 0.0
        ? 2.0 * result.boundary_precision * result.boundary_recall /
            (result.boundary_precision + result.boundary_recall)
        : 0.0;
    return result;
}

[[nodiscard]] Rlf5LanguageModelMetrics evaluate_language_model(
    const core::LanguageFabric& fabric,
    const std::string_view training_corpus,
    const std::string_view evaluation,
    const Rlf5Config& config,
    const CompositionalLanguageWorld& world
) {
    Rlf5LanguageModelMetrics result;
    const auto tokens = fabric.encode(evaluation);
    std::vector<std::uint64_t> history;
    history.reserve(fabric.config().maximum_context_order);
    double total_nll = 0.0;
    std::size_t correct = 0U;
    for (const auto token : tokens) {
        const auto prediction = fabric.predict_next(history);
        if (!prediction.outcomes.empty() &&
            prediction.outcomes.front().token_id == token) {
            ++correct;
        }
        double probability = probability_floor;
        const auto found = std::find_if(
            prediction.outcomes.begin(), prediction.outcomes.end(),
            [token](const core::LanguagePredictionOutcome& outcome) {
                return outcome.token_id == token;
            }
        );
        if (found != prediction.outcomes.end()) {
            probability = std::max(found->probability, probability_floor);
        }
        total_nll -= std::log(probability);
        ++result.predictions;
        if (fabric.lexeme_by_id(token).bytes.find('\n') != std::string::npos) {
            history.clear();
        } else {
            history.push_back(token);
            if (history.size() > fabric.config().maximum_context_order) {
                history.erase(history.begin());
            }
        }
    }
    result.top1_accuracy = result.predictions > 0U
        ? static_cast<double>(correct) / static_cast<double>(result.predictions)
        : 0.0;
    result.negative_log_likelihood = result.predictions > 0U
        ? total_nll / static_cast<double>(result.predictions)
        : 0.0;
    result.perplexity = std::exp(result.negative_log_likelihood);
    result.bits_per_byte = evaluation.empty()
        ? 0.0
        : total_nll / (ln2 * static_cast<double>(evaluation.size()));

    ByteNgram char_ngram(6U);
    char_ngram.train(training_corpus);
    result.char_ngram_bits_per_byte = char_ngram.bits_per_byte(evaluation);
    WordNgram word_ngram(4U);
    word_ngram.train(training_corpus);
    result.oracle_word_ngram_bits_per_byte = word_ngram.bits_per_byte(evaluation);

    core::DeterministicRng rng(config.seed ^ 0x4652454547454EULL);
    std::unordered_set<std::string> unique;
    std::size_t parseable = 0U;
    std::size_t total_bytes = 0U;
    for (std::size_t sample = 0U; sample < config.free_generation_samples; ++sample) {
        const auto heldout = world.evaluation_statements(1U, rng.next_u64()).front();
        const auto words = split_words(heldout.text);
        std::string prompt = "the ";
        if (words.size() >= 3U) {
            prompt += words[1U] + " " + words[2U];
        } else {
            prompt += "red fox";
        }
        const auto generated = fabric.generate(
            prompt, std::min<std::size_t>(48U, config.maximum_generation_tokens), true
        );
        unique.insert(generated);
        total_bytes += generated.size();
        if (fabric.parse(generated).success) {
            ++parseable;
        }
    }
    result.free_generation_parse_rate =
        static_cast<double>(parseable) /
        static_cast<double>(config.free_generation_samples);
    result.free_generation_unique_rate =
        static_cast<double>(unique.size()) /
        static_cast<double>(config.free_generation_samples);
    result.mean_generated_bytes =
        static_cast<double>(total_bytes) /
        static_cast<double>(config.free_generation_samples);
    return result;
}

[[nodiscard]] std::size_t frame_role_total(const core::LanguageFrame& frame) {
    std::size_t total = 0U;
    const std::array<std::string_view, 6U> values{
        frame.predicate, frame.agent, frame.patient, frame.agent_attribute,
        frame.patient_attribute, frame.location,
    };
    for (const auto value : values) {
        if (!value.empty()) {
            ++total;
        }
    }
    return total;
}

[[nodiscard]] std::size_t matching_roles(
    const core::LanguageFrame& left,
    const core::LanguageFrame& right
) {
    std::size_t matches = 0U;
    const std::array<std::pair<std::string_view, std::string_view>, 6U> values{{
        {left.predicate, right.predicate},
        {left.agent, right.agent},
        {left.patient, right.patient},
        {left.agent_attribute, right.agent_attribute},
        {left.patient_attribute, right.patient_attribute},
        {left.location, right.location},
    }};
    for (const auto& [expected, actual] : values) {
        if (!expected.empty() && expected == actual) {
            ++matches;
        }
    }
    return matches;
}

[[nodiscard]] Rlf5SemanticMetrics evaluate_semantics(
    const core::LanguageFabric& fabric,
    const std::span<const RenderedExample> evaluation,
    const NearestExampleBaseline& nearest,
    const BagOfWordsBaseline& bag
) {
    Rlf5SemanticMetrics result;
    result.examples = evaluation.size();
    std::size_t exact = 0U;
    std::size_t act = 0U;
    std::size_t roles = 0U;
    std::size_t total_roles = 0U;
    std::size_t coverage = 0U;
    std::size_t nearest_exact = 0U;
    std::size_t bag_exact = 0U;
    for (const auto& example : evaluation) {
        const auto parsed = fabric.parse(example.text);
        if (parsed.success) {
            ++coverage;
            if (parsed.frame.act == example.frame.act) {
                ++act;
            }
            roles += matching_roles(example.frame, parsed.frame);
            total_roles += frame_role_total(example.frame);
            if (parsed.frame == example.frame) {
                ++exact;
            }
        } else {
            total_roles += frame_role_total(example.frame);
        }
        if (nearest.predict(example.text) == example.frame) {
            ++nearest_exact;
        }
        if (bag.predict(example.text) == example.frame) {
            ++bag_exact;
        }
    }
    const double denominator = static_cast<double>(std::max<std::size_t>(1U, result.examples));
    result.frame_exact_accuracy = static_cast<double>(exact) / denominator;
    result.act_accuracy = static_cast<double>(act) / denominator;
    result.role_accuracy = total_roles > 0U
        ? static_cast<double>(roles) / static_cast<double>(total_roles)
        : 0.0;
    result.construction_coverage = static_cast<double>(coverage) / denominator;
    result.nearest_example_accuracy = static_cast<double>(nearest_exact) / denominator;
    result.bag_of_words_accuracy = static_cast<double>(bag_exact) / denominator;
    result.unseen_composition_accuracy = result.frame_exact_accuracy;
    return result;
}

[[nodiscard]] Rlf5GenerationMetrics evaluate_generation(
    const core::LanguageFabric& fabric,
    const std::span<const RenderedExample> evaluation,
    const std::unordered_set<std::string>& training_texts
) {
    Rlf5GenerationMetrics result;
    result.examples = evaluation.size();
    std::size_t successes = 0U;
    std::size_t roundtrip = 0U;
    std::size_t exact = 0U;
    std::size_t novel = 0U;
    std::unordered_set<std::uint64_t> distinct_tokens;
    std::size_t total_tokens = 0U;
    for (const auto& example : evaluation) {
        const auto generated = fabric.generate_frame(example.frame);
        if (!generated.success) {
            continue;
        }
        ++successes;
        if (generated.text == example.text) {
            ++exact;
        }
        if (!training_texts.contains(generated.text)) {
            ++novel;
        }
        for (const auto token : generated.tokens) {
            distinct_tokens.insert(token);
            ++total_tokens;
        }
        const auto parsed = fabric.parse(generated.text);
        if (parsed.success && parsed.frame == example.frame) {
            ++roundtrip;
        }
    }
    const double denominator = static_cast<double>(
        std::max<std::size_t>(1U, result.examples)
    );
    result.generation_success_rate = static_cast<double>(successes) / denominator;
    result.semantic_roundtrip_accuracy = static_cast<double>(roundtrip) / denominator;
    result.exact_surface_match_rate = static_cast<double>(exact) / denominator;
    result.novel_sentence_rate = static_cast<double>(novel) / denominator;
    result.lexical_diversity = total_tokens > 0U
        ? static_cast<double>(distinct_tokens.size()) /
            static_cast<double>(total_tokens)
        : 0.0;
    return result;
}

[[nodiscard]] Rlf5QuestionAnswerMetrics evaluate_qa(
    const core::LanguageFabric& fabric,
    const CompositionalLanguageWorld& world,
    const Rlf5Config& config
) {
    Rlf5QuestionAnswerMetrics result;
    result.episodes = config.qa_episodes;
    core::DeterministicRng rng(config.seed ^ 0x51414556414C35ULL);
    std::size_t success = 0U;
    std::size_t semantic = 0U;
    std::size_t roundtrip = 0U;
    std::size_t robust = 0U;
    std::size_t baseline = 0U;
    std::size_t question_parsed = 0U;
    std::size_t question_exact = 0U;
    std::size_t target_parsed = 0U;
    std::array<std::size_t, 3U> act_totals{};
    std::array<std::size_t, 3U> act_correct{};
    for (std::size_t index = 0U; index < config.qa_episodes; ++index) {
        const auto episode = world.qa_episode(rng);
        const auto query_parse = fabric.parse(episode.question);
        if (query_parse.success) {
            ++question_parsed;
            if (query_parse.frame == episode.query) {
                ++question_exact;
            }
        }
        const auto fact_parse = fabric.parse(episode.context[episode.target_index]);
        if (fact_parse.success && fact_parse.frame == episode.facts[episode.target_index]) {
            ++target_parsed;
        }
        const std::size_t act_index = static_cast<std::size_t>(episode.query.act) -
            static_cast<std::size_t>(core::LanguageAct::query_agent);
        if (act_index < act_totals.size()) {
            ++act_totals[act_index];
        }
        const auto answer = fabric.answer(episode.context, episode.question);
        if (answer.success) {
            ++success;
            if (answer.answer_frame == episode.answer) {
                ++semantic;
                if (act_index < act_correct.size()) {
                    ++act_correct[act_index];
                }
                if (answer.matched_fact == episode.facts[episode.target_index]) {
                    ++robust;
                }
            }
            const auto parsed = fabric.parse(answer.text);
            if (parsed.success && parsed.frame == episode.answer) {
                ++roundtrip;
            }
        }

        const auto question_words = split_words(episode.question);
        std::unordered_set<std::string> query(
            question_words.begin(), question_words.end()
        );
        std::size_t best_index = 0U;
        double best_score = -1.0;
        for (std::size_t fact_index = 0U; fact_index < episode.context.size(); ++fact_index) {
            const auto words = split_words(episode.context[fact_index]);
            std::unordered_set<std::string> candidate(words.begin(), words.end());
            std::size_t intersection = 0U;
            for (const auto& word : query) {
                if (candidate.contains(word)) {
                    ++intersection;
                }
            }
            const std::size_t union_size = query.size() + candidate.size() - intersection;
            const double score = union_size > 0U
                ? static_cast<double>(intersection) /
                    static_cast<double>(union_size)
                : 0.0;
            if (score > best_score) {
                best_score = score;
                best_index = fact_index;
            }
        }
        if (best_index == episode.target_index) {
            ++baseline;
        }
    }
    const double denominator = static_cast<double>(config.qa_episodes);
    result.answer_success_rate = static_cast<double>(success) / denominator;
    result.answer_semantic_accuracy = static_cast<double>(semantic) / denominator;
    result.answer_text_roundtrip_accuracy = static_cast<double>(roundtrip) / denominator;
    result.distractor_robustness = static_cast<double>(robust) / denominator;
    result.nearest_fact_baseline_accuracy = static_cast<double>(baseline) / denominator;
    result.question_parse_rate = static_cast<double>(question_parsed) / denominator;
    result.question_frame_exact_rate = static_cast<double>(question_exact) / denominator;
    result.target_fact_parse_rate = static_cast<double>(target_parsed) / denominator;
    result.agent_query_accuracy = act_totals[0U] > 0U
        ? static_cast<double>(act_correct[0U]) / static_cast<double>(act_totals[0U])
        : 0.0;
    result.patient_query_accuracy = act_totals[1U] > 0U
        ? static_cast<double>(act_correct[1U]) / static_cast<double>(act_totals[1U])
        : 0.0;
    result.location_query_accuracy = act_totals[2U] > 0U
        ? static_cast<double>(act_correct[2U]) / static_cast<double>(act_totals[2U])
        : 0.0;
    return result;
}

[[nodiscard]] Rlf5LeakageAudit audit_leakage(
    const std::string_view raw_corpus,
    const std::span<const core::LanguageSupervisedExample> training,
    const std::span<const RenderedExample> evaluation,
    const CompositionalLanguageWorld& world
) {
    Rlf5LeakageAudit audit;
    audit.raw_training_hash = world.corpus_hash(raw_corpus);
    std::uint64_t training_hash = fnv_offset_basis;
    std::unordered_set<std::string> training_texts;
    std::unordered_set<std::string> training_frames;
    for (const auto& example : training) {
        hash_string(training_hash, example.text);
        hash_string(training_hash, frame_key(example.frame));
        training_texts.insert(example.text);
        training_frames.insert(frame_key(example.frame));
    }
    audit.supervised_training_hash = training_hash;
    std::uint64_t evaluation_hash = fnv_offset_basis;
    for (const auto& example : evaluation) {
        hash_string(evaluation_hash, example.text);
        hash_string(evaluation_hash, frame_key(example.frame));
        if (training_texts.contains(example.text)) {
            ++audit.exact_text_overlap;
        }
        if (training_frames.contains(frame_key(example.frame))) {
            ++audit.exact_frame_overlap;
        }
    }
    audit.evaluation_hash = evaluation_hash;
    audit.evaluation_frames_withheld = audit.exact_frame_overlap == 0U;
    audit.exact_evaluation_text_withheld = audit.exact_text_overlap == 0U;
    audit.train_evaluation_hashes_disjoint =
        audit.supervised_training_hash != audit.evaluation_hash &&
        audit.raw_training_hash != audit.evaluation_hash;
    return audit;
}

struct EvaluationBundle final {
    Rlf5SegmentationMetrics segmentation;
    Rlf5LanguageModelMetrics language_model;
    Rlf5SemanticMetrics semantics;
    Rlf5GenerationMetrics generation;
    Rlf5QuestionAnswerMetrics qa;
    Rlf5LeakageAudit leakage;
};

[[nodiscard]] EvaluationBundle evaluate_all(
    const core::LanguageFabric& fabric,
    const CompositionalLanguageWorld& world,
    const Rlf5Config& config,
    const std::string_view raw_corpus,
    const std::span<const core::LanguageSupervisedExample> supervised,
    const std::span<const RenderedExample> evaluation
) {
    const std::string eval_corpus = evaluation_corpus(evaluation);
    const NearestExampleBaseline nearest(supervised);
    const BagOfWordsBaseline bag(supervised);
    std::unordered_set<std::string> training_texts;
    for (const auto& example : supervised) {
        training_texts.insert(example.text);
    }
    return {
        evaluate_segmentation(fabric, eval_corpus),
        evaluate_language_model(fabric, raw_corpus, eval_corpus, config, world),
        evaluate_semantics(fabric, evaluation, nearest, bag),
        evaluate_generation(fabric, evaluation, training_texts),
        evaluate_qa(fabric, world, config),
        audit_leakage(raw_corpus, supervised, evaluation, world),
    };
}

[[nodiscard]] Rlf5Result run_internal(
    const Rlf5Config& config,
    core::LanguageFabric* trained_output = nullptr
) {
    validate_config(config);
    const CompositionalLanguageWorld world(
        config, config.seed ^ 0x574F524C444C414EULL
    );
    const std::string raw = world.raw_corpus(config.raw_training_sentences);
    const auto supervised = world.supervised(config.supervised_training_examples);
    const auto evaluation = world.evaluation_statements(config.evaluation_examples);

    core::LanguageFabric fabric(fabric_config(config), config.seed);
    const auto start = std::chrono::steady_clock::now();
    fabric.learn_lexicon(raw);
    fabric.train_language_model(raw);
    fabric.train_semantics(supervised);
    const double training_seconds = std::chrono::duration<double>(
        std::chrono::steady_clock::now() - start
    ).count();

    const auto metrics = evaluate_all(
        fabric, world, config, raw, supervised, evaluation
    );
    Rlf5Result result;
    result.seed = config.seed;
    result.phase_dimension = config.phase_dimension;
    result.raw_training_sentences = config.raw_training_sentences;
    result.supervised_training_examples = config.supervised_training_examples;
    result.segmentation = metrics.segmentation;
    result.language_model = metrics.language_model;
    result.semantics = metrics.semantics;
    result.generation = metrics.generation;
    result.question_answering = metrics.qa;
    result.leakage_audit = metrics.leakage;
    result.training_stats = fabric.stats();
    result.learned_concepts = fabric.concepts().size();
    result.learned_constructions = fabric.constructions().size();
    result.estimated_model_bytes = fabric.estimated_storage_bytes();
    result.training_seconds = training_seconds;

    const bool leakage_passed = result.leakage_audit.evaluation_frames_withheld &&
        result.leakage_audit.exact_evaluation_text_withheld &&
        result.leakage_audit.train_evaluation_hashes_disjoint;
    const bool strong = leakage_passed &&
        result.semantics.frame_exact_accuracy >= 0.90 &&
        result.generation.semantic_roundtrip_accuracy >= 0.90 &&
        result.question_answering.answer_semantic_accuracy >= 0.90 &&
        result.language_model.free_generation_parse_rate >= 0.60 &&
        result.semantics.frame_exact_accuracy >=
            result.semantics.bag_of_words_accuracy + 0.05 &&
        result.semantics.frame_exact_accuracy >=
            result.semantics.nearest_example_accuracy + 0.20;
    const bool partial = leakage_passed &&
        result.semantics.frame_exact_accuracy >= 0.70 &&
        result.generation.semantic_roundtrip_accuracy >= 0.70 &&
        result.question_answering.answer_semantic_accuracy >= 0.70 &&
        result.semantics.frame_exact_accuracy >
            result.semantics.nearest_example_accuracy;
    result.scientific_decision = strong
        ? "A — strong controlled evidence"
        : partial
            ? "B — partial evidence"
            : "C — negative evidence";
    result.limitations = {
        "The language domain is deterministic controlled English rather than unrestricted web-scale language.",
        "Semantic grounding uses paired utterance-frame examples after raw self-supervised lexicon and sequence training.",
        "Learned constructions are compositional but remain a compact grammar family with fixed semantic roles.",
        "The benchmark does not establish broad world knowledge, multilingual competence, coding, or open-domain dialogue.",
        "Byte-pair lexicon learning and exact construction search remain CPU-bound and linear in stored structures.",
        "Free-running generation is evaluated for local grammatical/semantic validity, not long-document coherence.",
    };

    std::uint64_t hash = fabric.deterministic_hash();
    hash_u64(hash, result.leakage_audit.raw_training_hash);
    hash_u64(hash, result.leakage_audit.supervised_training_hash);
    hash_u64(hash, result.leakage_audit.evaluation_hash);
    hash_double(hash, result.segmentation.boundary_f1);
    hash_double(hash, result.language_model.top1_accuracy);
    hash_double(hash, result.semantics.frame_exact_accuracy);
    hash_double(hash, result.generation.semantic_roundtrip_accuracy);
    hash_double(hash, result.question_answering.answer_semantic_accuracy);
    hash_string(hash, result.scientific_decision);
    result.deterministic_run_hash = hash;
    if (trained_output != nullptr) {
        *trained_output = core::LanguageFabric::from_snapshot(fabric.snapshot());
    }
    return result;
}

void write_frame(std::ostream& output, const core::LanguageFrame& frame) {
    output << "{\"act\": \"" << core::language_act_name(frame.act)
        << "\", \"predicate\": \"" << json_escape(frame.predicate)
        << "\", \"agent\": \"" << json_escape(frame.agent)
        << "\", \"patient\": \"" << json_escape(frame.patient)
        << "\", \"agent_attribute\": \""
        << json_escape(frame.agent_attribute)
        << "\", \"patient_attribute\": \""
        << json_escape(frame.patient_attribute)
        << "\", \"location\": \"" << json_escape(frame.location) << "\"}";
}

}  // namespace

Rlf5Result run_rlf5_language(const Rlf5Config& config) {
    return run_internal(config);
}

void write_rlf5_result_json(std::ostream& output, const Rlf5Result& result) {
    output << std::setprecision(std::numeric_limits<double>::max_digits10)
        << "{\n  \"architecture\": \"RLF-5\",\n"
        << "  \"experiment\": \"compositional_language\",\n"
        << "  \"seed\": " << result.seed << ",\n"
        << "  \"phase_dimension\": " << result.phase_dimension << ",\n"
        << "  \"raw_training_sentences\": " << result.raw_training_sentences << ",\n"
        << "  \"supervised_training_examples\": "
        << result.supervised_training_examples << ",\n"
        << "  \"segmentation\": {\n"
        << "    \"raw_bytes\": " << result.segmentation.raw_bytes << ",\n"
        << "    \"encoded_tokens\": " << result.segmentation.encoded_tokens << ",\n"
        << "    \"learned_lexemes\": " << result.segmentation.learned_lexemes << ",\n"
        << "    \"learned_merges\": " << result.segmentation.learned_merges << ",\n"
        << "    \"compression_ratio\": " << result.segmentation.compression_ratio << ",\n"
        << "    \"boundary_precision\": " << result.segmentation.boundary_precision << ",\n"
        << "    \"boundary_recall\": " << result.segmentation.boundary_recall << ",\n"
        << "    \"boundary_f1\": " << result.segmentation.boundary_f1 << ",\n"
        << "    \"exact_roundtrip_rate\": " << result.segmentation.exact_roundtrip_rate << "\n"
        << "  },\n  \"language_model\": {\n"
        << "    \"predictions\": " << result.language_model.predictions << ",\n"
        << "    \"top1_accuracy\": " << result.language_model.top1_accuracy << ",\n"
        << "    \"negative_log_likelihood\": "
        << result.language_model.negative_log_likelihood << ",\n"
        << "    \"perplexity\": " << result.language_model.perplexity << ",\n"
        << "    \"bits_per_byte\": " << result.language_model.bits_per_byte << ",\n"
        << "    \"char_ngram_bits_per_byte\": "
        << result.language_model.char_ngram_bits_per_byte << ",\n"
        << "    \"oracle_word_ngram_bits_per_byte\": "
        << result.language_model.oracle_word_ngram_bits_per_byte << ",\n"
        << "    \"free_generation_parse_rate\": "
        << result.language_model.free_generation_parse_rate << ",\n"
        << "    \"free_generation_unique_rate\": "
        << result.language_model.free_generation_unique_rate << ",\n"
        << "    \"mean_generated_bytes\": "
        << result.language_model.mean_generated_bytes << "\n"
        << "  },\n  \"semantics\": {\n"
        << "    \"examples\": " << result.semantics.examples << ",\n"
        << "    \"frame_exact_accuracy\": "
        << result.semantics.frame_exact_accuracy << ",\n"
        << "    \"act_accuracy\": " << result.semantics.act_accuracy << ",\n"
        << "    \"role_accuracy\": " << result.semantics.role_accuracy << ",\n"
        << "    \"construction_coverage\": "
        << result.semantics.construction_coverage << ",\n"
        << "    \"nearest_example_accuracy\": "
        << result.semantics.nearest_example_accuracy << ",\n"
        << "    \"bag_of_words_accuracy\": "
        << result.semantics.bag_of_words_accuracy << ",\n"
        << "    \"unseen_composition_accuracy\": "
        << result.semantics.unseen_composition_accuracy << "\n"
        << "  },\n  \"generation\": {\n"
        << "    \"examples\": " << result.generation.examples << ",\n"
        << "    \"generation_success_rate\": "
        << result.generation.generation_success_rate << ",\n"
        << "    \"semantic_roundtrip_accuracy\": "
        << result.generation.semantic_roundtrip_accuracy << ",\n"
        << "    \"exact_surface_match_rate\": "
        << result.generation.exact_surface_match_rate << ",\n"
        << "    \"novel_sentence_rate\": "
        << result.generation.novel_sentence_rate << ",\n"
        << "    \"lexical_diversity\": "
        << result.generation.lexical_diversity << "\n"
        << "  },\n  \"question_answering\": {\n"
        << "    \"episodes\": " << result.question_answering.episodes << ",\n"
        << "    \"answer_success_rate\": "
        << result.question_answering.answer_success_rate << ",\n"
        << "    \"answer_semantic_accuracy\": "
        << result.question_answering.answer_semantic_accuracy << ",\n"
        << "    \"answer_text_roundtrip_accuracy\": "
        << result.question_answering.answer_text_roundtrip_accuracy << ",\n"
        << "    \"distractor_robustness\": "
        << result.question_answering.distractor_robustness << ",\n"
        << "    \"nearest_fact_baseline_accuracy\": "
        << result.question_answering.nearest_fact_baseline_accuracy << ",\n"
        << "    \"question_parse_rate\": "
        << result.question_answering.question_parse_rate << ",\n"
        << "    \"question_frame_exact_rate\": "
        << result.question_answering.question_frame_exact_rate << ",\n"
        << "    \"target_fact_parse_rate\": "
        << result.question_answering.target_fact_parse_rate << ",\n"
        << "    \"agent_query_accuracy\": "
        << result.question_answering.agent_query_accuracy << ",\n"
        << "    \"patient_query_accuracy\": "
        << result.question_answering.patient_query_accuracy << ",\n"
        << "    \"location_query_accuracy\": "
        << result.question_answering.location_query_accuracy << "\n"
        << "  },\n  \"leakage_audit\": {\n"
        << "    \"raw_corpus_contains_no_semantic_labels\": "
        << (result.leakage_audit.raw_corpus_contains_no_semantic_labels ? "true" : "false") << ",\n"
        << "    \"evaluation_frames_withheld\": "
        << (result.leakage_audit.evaluation_frames_withheld ? "true" : "false") << ",\n"
        << "    \"exact_evaluation_text_withheld\": "
        << (result.leakage_audit.exact_evaluation_text_withheld ? "true" : "false") << ",\n"
        << "    \"train_evaluation_hashes_disjoint\": "
        << (result.leakage_audit.train_evaluation_hashes_disjoint ? "true" : "false") << ",\n"
        << "    \"exact_text_overlap\": " << result.leakage_audit.exact_text_overlap << ",\n"
        << "    \"exact_frame_overlap\": " << result.leakage_audit.exact_frame_overlap << ",\n"
        << "    \"raw_training_hash\": \""
        << format_hash(result.leakage_audit.raw_training_hash) << "\",\n"
        << "    \"supervised_training_hash\": \""
        << format_hash(result.leakage_audit.supervised_training_hash) << "\",\n"
        << "    \"evaluation_hash\": \""
        << format_hash(result.leakage_audit.evaluation_hash) << "\"\n"
        << "  },\n"
        << "  \"learned_concepts\": " << result.learned_concepts << ",\n"
        << "  \"learned_constructions\": " << result.learned_constructions << ",\n"
        << "  \"estimated_model_bytes\": " << result.estimated_model_bytes << ",\n"
        << "  \"training_seconds\": " << result.training_seconds << ",\n"
        << "  \"deterministic_run_hash\": \""
        << format_hash(result.deterministic_run_hash) << "\",\n"
        << "  \"scientific_decision\": \""
        << json_escape(result.scientific_decision) << "\",\n"
        << "  \"limitations\": [\n";
    for (std::size_t index = 0U; index < result.limitations.size(); ++index) {
        output << "    \"" << json_escape(result.limitations[index]) << "\"";
        if (index + 1U != result.limitations.size()) {
            output << ',';
        }
        output << '\n';
    }
    output << "  ]\n}\n";
}

Rlf5TrainingWorkflowResult train_rlf5_checkpoint(
    const Rlf5Config& config,
    const std::filesystem::path& checkpoint_path
) {
    core::LanguageFabric trained(fabric_config(config), config.seed);
    const auto result = run_internal(config, &trained);
    storage::save_rlf5_checkpoint(checkpoint_path, trained);
    return {
        checkpoint_path,
        config.seed,
        config.raw_training_sentences,
        trained.lexemes().size(),
        trained.contexts().size(),
        trained.concepts().size(),
        trained.constructions().size(),
        result.deterministic_run_hash,
    };
}

Rlf5EvaluationWorkflowResult evaluate_rlf5_checkpoint(
    const std::filesystem::path& checkpoint_path,
    const std::uint64_t seed,
    const std::size_t evaluation_examples
) {
    auto fabric = storage::load_rlf5_checkpoint(checkpoint_path);
    Rlf5Config config;
    config.seed = seed;
    config.phase_dimension = fabric.config().phase_dimension;
    config.evaluation_examples = std::max<std::size_t>(64U, evaluation_examples);
    config.qa_episodes = std::max<std::size_t>(32U, evaluation_examples / 3U);
    config.free_generation_samples = std::min<std::size_t>(64U, evaluation_examples);
    config.holdout_modulus = 7U;
    const CompositionalLanguageWorld world(
        config, fabric.seed() ^ 0x574F524C444C414EULL
    );
    const auto evaluation = world.evaluation_statements(config.evaluation_examples);
    const auto synthetic_supervised = world.supervised(512U);
    const NearestExampleBaseline nearest(synthetic_supervised);
    const BagOfWordsBaseline bag(synthetic_supervised);
    const auto semantics = evaluate_semantics(fabric, evaluation, nearest, bag);
    std::unordered_set<std::string> none;
    const auto generation = evaluate_generation(fabric, evaluation, none);
    const auto qa = evaluate_qa(fabric, world, config);
    const std::string corpus = evaluation_corpus(evaluation);
    std::uint64_t hash = fabric.deterministic_hash();
    hash_double(hash, semantics.frame_exact_accuracy);
    hash_double(hash, generation.semantic_roundtrip_accuracy);
    hash_double(hash, qa.answer_semantic_accuracy);
    return {
        checkpoint_path,
        semantics,
        generation,
        qa,
        fabric.sequence_nll(corpus) *
            static_cast<double>(fabric.encode(corpus).size()) /
            (ln2 * static_cast<double>(std::max<std::size_t>(1U, corpus.size()))),
        hash,
    };
}

Rlf5TraceWorkflowResult trace_rlf5_checkpoint(
    const std::filesystem::path& checkpoint_path,
    const std::uint64_t seed,
    const std::size_t sample_id
) {
    auto fabric = storage::load_rlf5_checkpoint(checkpoint_path);
    Rlf5Config config;
    config.seed = seed;
    config.phase_dimension = fabric.config().phase_dimension;
    const CompositionalLanguageWorld world(
        config, fabric.seed() ^ 0x574F524C444C414EULL
    );
    RenderedExample statement;
    core::LanguageParse parse;
    std::size_t selected_sample = sample_id;
    for (std::size_t attempt = 0U; attempt < 64U; ++attempt) {
        selected_sample = sample_id + attempt;
        statement = world.evaluation_statements(1U, selected_sample).front();
        parse = fabric.parse(statement.text);
        if (parse.success) {
            break;
        }
    }
    core::DeterministicRng rng(seed ^ selected_sample ^ 0x545241434535ULL);
    CompositionalLanguageWorld::QaEpisode episode;
    core::LanguageParse question_parse;
    core::LanguageAnswer answer;
    for (std::size_t attempt = 0U; attempt < 64U; ++attempt) {
        episode = world.qa_episode(rng);
        question_parse = fabric.parse(episode.question);
        answer = fabric.answer(episode.context, episode.question);
        if (question_parse.success && answer.success) {
            break;
        }
    }
    const auto generated = fabric.generate_frame(statement.frame);
    return {
        checkpoint_path,
        selected_sample,
        statement.text,
        fabric.encode(statement.text),
        parse,
        generated.text,
        episode.question,
        question_parse,
        answer,
    };
}

void write_rlf5_training_json(
    std::ostream& output,
    const Rlf5TrainingWorkflowResult& result
) {
    output << "{\n  \"architecture\": \"RLF-5\",\n"
        << "  \"checkpoint\": \"" << json_escape(result.checkpoint_path.string()) << "\",\n"
        << "  \"seed\": " << result.seed << ",\n"
        << "  \"raw_training_sentences\": " << result.raw_training_sentences << ",\n"
        << "  \"lexemes\": " << result.lexemes << ",\n"
        << "  \"contexts\": " << result.contexts << ",\n"
        << "  \"concepts\": " << result.concepts << ",\n"
        << "  \"constructions\": " << result.constructions << ",\n"
        << "  \"deterministic_run_hash\": \""
        << format_hash(result.deterministic_run_hash) << "\"\n}\n";
}

void write_rlf5_evaluation_json(
    std::ostream& output,
    const Rlf5EvaluationWorkflowResult& result
) {
    output << std::setprecision(std::numeric_limits<double>::max_digits10)
        << "{\n  \"architecture\": \"RLF-5\",\n"
        << "  \"checkpoint\": \"" << json_escape(result.checkpoint_path.string()) << "\",\n"
        << "  \"frame_exact_accuracy\": "
        << result.semantics.frame_exact_accuracy << ",\n"
        << "  \"semantic_roundtrip_accuracy\": "
        << result.generation.semantic_roundtrip_accuracy << ",\n"
        << "  \"answer_semantic_accuracy\": "
        << result.question_answering.answer_semantic_accuracy << ",\n"
        << "  \"language_bits_per_byte\": " << result.language_bits_per_byte << ",\n"
        << "  \"deterministic_run_hash\": \""
        << format_hash(result.deterministic_run_hash) << "\"\n}\n";
}

void write_rlf5_trace_json(
    std::ostream& output,
    const Rlf5TraceWorkflowResult& result
) {
    output << std::setprecision(std::numeric_limits<double>::max_digits10)
        << "{\n  \"architecture\": \"RLF-5\",\n"
        << "  \"checkpoint\": \"" << json_escape(result.checkpoint_path.string()) << "\",\n"
        << "  \"sample_id\": " << result.sample_id << ",\n"
        << "  \"statement\": \"" << json_escape(result.statement) << "\",\n"
        << "  \"statement_tokens\": [";
    for (std::size_t index = 0U; index < result.statement_tokens.size(); ++index) {
        if (index != 0U) { output << ", "; }
        output << result.statement_tokens[index];
    }
    output << "],\n  \"statement_parse\": ";
    write_frame(output, result.statement_parse.frame);
    output << ",\n  \"statement_parse_success\": "
        << (result.statement_parse.success ? "true" : "false") << ",\n"
        << "  \"generated_statement\": \""
        << json_escape(result.generated_statement) << "\",\n"
        << "  \"question\": \"" << json_escape(result.question) << "\",\n"
        << "  \"question_parse\": ";
    write_frame(output, result.question_parse.frame);
    output << ",\n  \"answer_success\": "
        << (result.answer.success ? "true" : "false") << ",\n"
        << "  \"answer_text\": \"" << json_escape(result.answer.text) << "\",\n"
        << "  \"answer_frame\": ";
    write_frame(output, result.answer.answer_frame);
    output << "\n}\n";
}

}  // namespace rlf::experiments
