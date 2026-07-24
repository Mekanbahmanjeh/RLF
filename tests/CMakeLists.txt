add_executable(rlf_tests
    test_main.cpp
    test_deterministic_rng.cpp
    test_data_pipeline.cpp
    test_evaluation_runner.cpp
    test_associative_memory.cpp
    test_associative_recall.cpp
    test_backends.cpp
    test_baselines.cpp
    test_checkpoint.cpp
    test_checkpoint_workflow.cpp
    test_rlf1_checkpoint.cpp
    test_rlf2_checkpoint.cpp
    test_rlf3_checkpoint.cpp
    test_rlf4_checkpoint.cpp
    test_rlf5_checkpoint.cpp
    test_rlf5_language.cpp
    test_rlf6_agent.cpp
    test_rlf6_checkpoint.cpp
    test_rlf7_frontier.cpp
    test_capacity_scaling.cpp
    test_compositional_generalization.cpp
    test_continual_learning.cpp
    test_encoding.cpp
    test_local_learning.cpp
    test_latent_routing.cpp
    test_language_fabric.cpp
    test_operator_fabric.cpp
    test_operator_composition.cpp
    test_optimization_benchmark.cpp
    test_rlf1_latent_routing.cpp
    test_rlf2_predictive_reasoning.cpp
    test_rlf3_world_model.cpp
    test_rlf4_self_supervised.cpp
    test_phase_vector.cpp
    test_predictive_skill_fabric.cpp
    test_quantized_phase_vector.cpp
    test_phase_vector_smoke.cpp
    test_persistence_roundtrip.cpp
    test_resonant_fabric.cpp
    test_retrieval.cpp
    test_sequence_completion.cpp
    test_structural_learning.cpp
    test_sparse_world_model.cpp
    test_temporal_predictive_fabric.cpp
    test_structural_adaptation.cpp
    test_transformation_learning.cpp
    test_solstice.cpp
    test_frontier_research.cpp
    test_efficiency_proof.cpp
    test_efficiency_campaign.cpp
    test_general_frontier.cpp
    test_image_generation_fabric.cpp
    test_image_generation_checkpoint.cpp
    test_image_generation_artifact_manifest.cpp
    test_image_generation_data_audit.cpp
    test_resonant_image_fabric.cpp
    test_prompt_semantic_fabric.cpp
)
target_link_libraries(rlf_tests PRIVATE rlf::core)
target_include_directories(rlf_tests PRIVATE "${CMAKE_CURRENT_SOURCE_DIR}")

add_test(NAME rlf_tests COMMAND rlf_tests)

add_executable(rlf6_tests
    test_main.cpp
    test_rlf6_agent.cpp
    test_rlf6_checkpoint.cpp
)
target_link_libraries(rlf6_tests PRIVATE rlf::core)
target_include_directories(rlf6_tests PRIVATE "${CMAKE_CURRENT_SOURCE_DIR}")

add_executable(rlf7_frontier_tests
    test_main.cpp
    test_rlf7_frontier.cpp
)
target_link_libraries(rlf7_frontier_tests PRIVATE rlf::core)
target_include_directories(rlf7_frontier_tests PRIVATE "${CMAKE_CURRENT_SOURCE_DIR}")
add_test(NAME rlf7_frontier_tests COMMAND rlf7_frontier_tests)

add_executable(solstice_tests
    test_main.cpp
    test_solstice.cpp
    test_frontier_research.cpp
    test_efficiency_proof.cpp
    test_efficiency_campaign.cpp
    test_general_frontier.cpp
    test_image_generation_fabric.cpp
    test_image_generation_checkpoint.cpp
    test_image_generation_artifact_manifest.cpp
    test_image_generation_data_audit.cpp
    test_resonant_image_fabric.cpp
    test_prompt_semantic_fabric.cpp
    test_data_pipeline.cpp
    test_evaluation_runner.cpp
)
target_link_libraries(solstice_tests PRIVATE rlf::core)
target_include_directories(solstice_tests PRIVATE "${CMAKE_CURRENT_SOURCE_DIR}")
add_test(NAME solstice_tests COMMAND solstice_tests)

if(UNIX)
    add_test(
        NAME h100_vram_guard_tests
        COMMAND bash "${CMAKE_CURRENT_SOURCE_DIR}/test_h100_vram_guard.sh"
    )
    add_test(
        NAME h200_vram_guard_tests
        COMMAND bash "${CMAKE_CURRENT_SOURCE_DIR}/test_h200_vram_guard.sh"
    )
    add_test(
        NAME training_artifact_manifest_tests
        COMMAND bash "${CMAKE_CURRENT_SOURCE_DIR}/test_training_artifact_manifest.sh"
            "$<TARGET_FILE:rlf_general_benchmark>"
    )
    add_test(
        NAME general_h100_readiness_tests
        COMMAND bash "${CMAKE_CURRENT_SOURCE_DIR}/test_general_h100_readiness.sh"
    )
    add_test(
        NAME general_h200_readiness_tests
        COMMAND bash "${CMAKE_CURRENT_SOURCE_DIR}/test_general_h200_readiness.sh"
    )
    add_test(
        NAME general_cuda_vram_guard_tests
        COMMAND bash "${CMAKE_CURRENT_SOURCE_DIR}/test_general_cuda_vram_guard.sh"
    )
    add_test(
        NAME general_cuda_readiness_tests
        COMMAND bash "${CMAKE_CURRENT_SOURCE_DIR}/test_general_cuda_readiness.sh"
    )
    add_test(
        NAME general_cuda_artifact_manifest_tests
        COMMAND bash "${CMAKE_CURRENT_SOURCE_DIR}/test_general_cuda_artifact_manifest.sh"
    )
    add_test(
        NAME general_cuda_profile_enforcement_tests
        COMMAND bash "${CMAKE_CURRENT_SOURCE_DIR}/test_general_cuda_profile_enforcement.sh"
            "$<TARGET_FILE:solstice>"
    )
    add_test(
        NAME general_cuda_training_handoff_tests
        COMMAND bash "${CMAKE_CURRENT_SOURCE_DIR}/test_general_cuda_training_handoff.sh"
            "$<TARGET_FILE:solstice>"
    )
    add_test(
        NAME efficiency_baseline_harness_tests
        COMMAND bash "${CMAKE_CURRENT_SOURCE_DIR}/test_efficiency_baseline_harness.sh"
    )
    add_test(
        NAME wall_budget_tests
        COMMAND bash "${CMAKE_CURRENT_SOURCE_DIR}/test_wall_budget.sh"
    )
    add_test(
        NAME checkpoint_transaction_tests
        COMMAND bash "${CMAKE_CURRENT_SOURCE_DIR}/test_checkpoint_transaction.sh"
    )
    add_test(
        NAME training_data_layout_tests
        COMMAND bash "${CMAKE_CURRENT_SOURCE_DIR}/test_training_data_layout.sh"
    )
    add_test(
        NAME vast_phase1_helper_tests
        COMMAND bash "${CMAKE_CURRENT_SOURCE_DIR}/test_vast_phase1_helpers.sh"
    )
    add_test(
        NAME v100_1586h_campaign_tests
        COMMAND bash "${CMAKE_CURRENT_SOURCE_DIR}/test_v100_1586h_campaign.sh"
    )
    add_test(
        NAME h200_12month_campaign_tests
        COMMAND bash "${CMAKE_CURRENT_SOURCE_DIR}/test_h200_12month_campaign.sh"
    )
    add_test(
        NAME imagegen_cli_tests
        COMMAND bash "${CMAKE_CURRENT_SOURCE_DIR}/test_imagegen_cli.sh"
            "$<TARGET_FILE:solstice>"
    )
    add_test(
        NAME imagegen_v100_readiness_tests
        COMMAND bash "${CMAKE_CURRENT_SOURCE_DIR}/test_imagegen_v100_readiness.sh"
    )
    add_test(
        NAME imagegen_data_audit_tests
        COMMAND bash "${CMAKE_CURRENT_SOURCE_DIR}/test_imagegen_data_audits.sh"
    )
    add_test(
        NAME imagegen_transactional_attempt_tests
        COMMAND bash "${CMAKE_CURRENT_SOURCE_DIR}/test_imagegen_transactional_attempt.sh"
    )
    add_test(
        NAME imagegen_40h_controller_tests
        COMMAND bash "${CMAKE_CURRENT_SOURCE_DIR}/test_imagegen_40h_controller.sh"
    )
    add_test(
        NAME imagegen_external_diffusion_baseline_tests
        COMMAND bash "${CMAKE_CURRENT_SOURCE_DIR}/test_imagegen_external_diffusion_baseline.sh"
    )
    add_test(
        NAME imagegen_40h_result_gate_tests
        COMMAND bash "${CMAKE_CURRENT_SOURCE_DIR}/test_imagegen_40h_result_gate.sh"
    )
    add_test(
        NAME imagegen_5k_dataset_contract_tests
        COMMAND bash "${CMAKE_CURRENT_SOURCE_DIR}/test_imagegen_5k_dataset_contract.sh"
            "$<TARGET_FILE:solstice>"
    )
endif()

add_executable(arc_agi2_adapter_tests
    test_main.cpp
    test_arc_agi2.cpp
)
target_link_libraries(arc_agi2_adapter_tests PRIVATE rlf::arc_agi2_adapter)
target_include_directories(arc_agi2_adapter_tests PRIVATE "${CMAKE_CURRENT_SOURCE_DIR}")
add_test(NAME arc_agi2_adapter_tests COMMAND arc_agi2_adapter_tests)
