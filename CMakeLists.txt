cmake_minimum_required(VERSION 3.25)

project(
    RLF
    VERSION 11.0.0
    DESCRIPTION "RLF General Frontier Attempt with semantic instruction memory, deliberation, preferences, multimodal grounding, tools, and auditable benchmark gates"
    LANGUAGES CXX
)

option(RLF_BUILD_TESTS "Build the RLF test suite" ON)
option(RLF_ENABLE_ASAN "Enable AddressSanitizer" OFF)
option(RLF_ENABLE_UBSAN "Enable UndefinedBehaviorSanitizer" OFF)
option(RLF_WARNINGS_AS_ERRORS "Treat compiler warnings as errors" ON)
option(RLF_ENABLE_CUDA "Build the RLF-Frontier CUDA backend" OFF)
option(RLF_ENABLE_SYSTEM_IMAGE_CODECS "Use libpng/libjpeg when available" ON)

set(CMAKE_CXX_STANDARD 23)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

list(APPEND CMAKE_MODULE_PATH "${CMAKE_CURRENT_SOURCE_DIR}/cmake")
include(CompilerWarnings)
include(Sanitizers)

add_library(rlf_project_options INTERFACE)
rlf_set_project_warnings(rlf_project_options)
rlf_enable_sanitizers(rlf_project_options)

add_library(rlf_core
    src/agent/agent_fabric.cpp
    src/backend/compute_backend.cpp
    src/baselines/nearest_neighbor.cpp
    src/baselines/prototype_classifier.cpp
    src/baselines/transition_table.cpp
    src/core/deterministic_rng.cpp
    src/core/encoding.cpp
    src/core/latent_routing.cpp
    src/core/language_fabric.cpp
    src/core/phase_vector.cpp
    src/core/predictive_skill_fabric.cpp
    src/core/quantized_phase_vector.cpp
    src/core/operator_fabric.cpp
    src/core/resonant_fabric.cpp
    src/core/resonant_mode.cpp
    src/core/sha256.cpp
    src/core/sparse_world_model.cpp
    src/core/temporal_predictive_fabric.cpp
    src/core/settling.cpp
    src/core/transformation_operator.cpp
    src/experiments/associative_recall.cpp
    src/experiments/capacity_scaling.cpp
    src/experiments/checkpoint_workflow.cpp
    src/experiments/compositional_generalization.cpp
    src/experiments/continual_learning.cpp
    src/experiments/persistence_roundtrip.cpp
    src/experiments/phase_vector_smoke.cpp
    src/experiments/metrics.cpp
    src/experiments/operator_composition.cpp
    src/experiments/optimization_benchmark.cpp
    src/experiments/rlf1_latent_routing.cpp
    src/experiments/rlf2_predictive_reasoning.cpp
    src/experiments/rlf3_world_model.cpp
    src/experiments/rlf4_self_supervised.cpp
    src/experiments/rlf5_language.cpp
    src/experiments/rlf6_agent.cpp
    src/experiments/sequence_completion.cpp
    src/experiments/structural_adaptation.cpp
    src/experiments/transformation_task_suite.cpp
    src/experiments/transformation_learning.cpp
    src/experiments/rlf7_frontier.cpp
    src/frontier/knowledge_fabric.cpp
    src/frontier/multimodal.cpp
    src/frontier/frontier_trainer.cpp
    src/frontier/frontier_backend.cpp
    src/learning/local_learning.cpp
    src/learning/structural_learning.cpp
    src/memory/associative_memory.cpp
    src/retrieval/exact_mode_retriever.cpp
    src/retrieval/contiguous_mode_index.cpp
    src/storage/checkpoint.cpp
    src/storage/rlf1_checkpoint.cpp
    src/storage/rlf2_checkpoint.cpp
    src/storage/rlf3_checkpoint.cpp
    src/storage/rlf4_checkpoint.cpp
    src/storage/rlf5_checkpoint.cpp
    src/storage/rlf6_checkpoint.cpp
    src/storage/rlf7_checkpoint.cpp
    src/solstice/tokenizer.cpp
    src/solstice/sparse_router.cpp
    src/solstice/abstraction_fabric.cpp
    src/solstice/continual_learning.cpp
    src/solstice/data_pipeline.cpp
    src/solstice/evaluation_runner.cpp
    src/solstice/grounding_fabric.cpp
    src/solstice/general_fabric.cpp
    src/solstice/frontier_gate.cpp
    src/solstice/efficiency_proof.cpp
    src/solstice/efficiency_campaign.cpp
    src/solstice/efficiency_corpus.cpp
    src/solstice/profile.cpp
    src/solstice/image_generation_fabric.cpp
    src/solstice/image_generation_evidence.cpp
    src/solstice/image_generation_checkpoint.cpp
    src/solstice/image_generation_artifact_manifest.cpp
    src/solstice/image_generation_data_audit.cpp
    src/solstice/prompt_semantic_fabric.cpp
    src/solstice/resonant_image_fabric.cpp
    src/solstice/language_fabric.cpp
    src/solstice/vision_fabric.cpp
    src/solstice/video_fabric.cpp
    src/solstice/tool_protocol.cpp
    src/solstice/solstice_model.cpp
    src/solstice/checkpoint.cpp
)
add_library(rlf::core ALIAS rlf_core)

if(RLF_ENABLE_CUDA)
    enable_language(CUDA)
    find_package(CUDAToolkit REQUIRED)
    target_sources(rlf_core PRIVATE src/frontier/frontier_cuda.cu)
    target_compile_definitions(rlf_core PUBLIC RLF_HAS_CUDA=1)
    set_target_properties(rlf_core PROPERTIES CUDA_STANDARD 20 CUDA_STANDARD_REQUIRED ON)
    target_link_libraries(rlf_core PUBLIC CUDA::cudart)
endif()

if(RLF_ENABLE_SYSTEM_IMAGE_CODECS)
    find_package(PNG QUIET)
    if(PNG_FOUND)
        target_link_libraries(rlf_core PUBLIC PNG::PNG)
        target_compile_definitions(rlf_core PUBLIC RLF_HAS_LIBPNG=1)
    endif()

    find_package(JPEG QUIET)
    if(JPEG_FOUND)
        target_link_libraries(rlf_core PUBLIC JPEG::JPEG)
        target_compile_definitions(rlf_core PUBLIC RLF_HAS_LIBJPEG=1)
    endif()
endif()

target_include_directories(
    rlf_core
    PUBLIC
        "$<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>"
        "$<INSTALL_INTERFACE:include>"
    PRIVATE
        "${CMAKE_CURRENT_SOURCE_DIR}/src"
)
target_link_libraries(rlf_core PUBLIC rlf_project_options)

add_library(rlf_arc_agi2_adapter src/benchmarks/arc_agi2.cpp)
add_library(rlf::arc_agi2_adapter ALIAS rlf_arc_agi2_adapter)
target_include_directories(
    rlf_arc_agi2_adapter
    PUBLIC "$<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>"
)
target_link_libraries(rlf_arc_agi2_adapter PUBLIC rlf::core)

add_executable(rlf src/cli/main.cpp)
target_link_libraries(rlf PRIVATE rlf::core)

add_executable(solstice src/solstice/cli.cpp)
target_link_libraries(solstice PRIVATE rlf::core)

add_executable(rlf_frontier_benchmark src/solstice/frontier_benchmark.cpp)
target_link_libraries(rlf_frontier_benchmark PRIVATE rlf::core)

add_executable(rlf_efficiency_proof src/solstice/efficiency_proof_cli.cpp)
target_link_libraries(rlf_efficiency_proof PRIVATE rlf::core)

add_executable(rlf_efficiency_campaign src/solstice/efficiency_campaign_cli.cpp)
target_link_libraries(rlf_efficiency_campaign PRIVATE rlf::core)

add_executable(rlf_general_benchmark src/solstice/general_benchmark_cli.cpp)
target_link_libraries(rlf_general_benchmark PRIVATE rlf::core)

add_executable(rlf_arc_agi2 src/benchmarks/arc_agi2_cli.cpp)
target_link_libraries(rlf_arc_agi2 PRIVATE rlf::arc_agi2_adapter)

if(WIN32)
    target_link_libraries(rlf_core PUBLIC windowscodecs ole32)
endif()

include(GNUInstallDirs)
install(TARGETS rlf solstice rlf_frontier_benchmark rlf_efficiency_proof rlf_efficiency_campaign rlf_general_benchmark rlf_arc_agi2 RUNTIME DESTINATION "${CMAKE_INSTALL_BINDIR}")
install(DIRECTORY include/ DESTINATION "${CMAKE_INSTALL_INCLUDEDIR}")

if(RLF_BUILD_TESTS)
    include(CTest)
    enable_testing()
    add_subdirectory(tests)
endif()
