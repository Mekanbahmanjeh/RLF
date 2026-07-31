# RLF SDK

The RLF SDK is the stable application layer over the research components. It provides a
Transformers-style local loading and pipeline API while retaining RLF-specific guarantees:
checkpoint integrity, optional profile enforcement, declared task support, deterministic defaults,
and no implicit network access.

## Model bundle

An RLF model bundle is a directory containing a checkpoint and `rlf-bundle.conf`:

```ini
format=1
name=solstice-preview
architecture=solstice
checkpoint=model.rlfsp
checkpoint_sha256=<64 lowercase hexadecimal characters>
profile=preview-6g
tasks=text-generation,image-text-to-text,tool-use
license=MIT
```

The loader rejects unknown fields, duplicate fields, unsupported formats and architectures,
absolute or escaping checkpoint paths, malformed hashes, hash mismatches, unsupported tasks, and
profile/checkpoint mismatches. A raw `.rlfsp` checkpoint can also be loaded directly.

## Context windows

RLF does not use a Transformer KV cache, so a single "context window" number would be misleading.
The SDK reports four separate limits:

- `maximum_predictive_context_tokens`: longest exact trailing token order used for next-token
  prediction.
- `maximum_episode_cue_tokens`: token budget used when matching learned dialogue episodes.
- `maximum_generation_tokens`: hard model-side generation ceiling.
- `maximum_retrieval_context_characters`: bounded context assembled from grounding, knowledge, and
  retrieved instruction demonstrations.

The `preview-6g` profile currently reports 64 predictive tokens, 256 episodic cue tokens, 256
maximum generated tokens, and 32,768 retrieval-context characters. These are different mechanisms
and must not be added together as if they were one Transformer context length.

Stateful multi-turn use is provided by `ChatSession`. It serializes recent complete turns, measures
them with the model tokenizer, evicts oldest turns first, optionally evicts the system prompt, and
truncates only the current prompt as a last resort:

```cpp
#include <rlf/sdk/session.hpp>

rlf::sdk::ChatSession chat(
    rlf::sdk::make_pipeline(
        rlf::sdk::PipelineTask::text_generation,
        "/models/solstice-preview"
    ),
    {
        .maximum_context_tokens = 256,
        .maximum_turns = 16,
        .system_prompt = "Answer only from learned evidence.",
    }
);

const auto answer = chat.send("What did we discuss previously?");
const auto usage = chat.context_stats();
```

## C++ usage

```cpp
#include <rlf/sdk/pipeline.hpp>

const auto model = rlf::sdk::AutoModel::from_pretrained(
    "/models/solstice-preview",
    {.backend = rlf::frontier::FrontierBackendKind::optimized_cpu}
);

const rlf::sdk::Pipeline generate(
    rlf::sdk::PipelineTask::text_generation,
    model
);
const auto answer = generate("What can you do?");
```

The one-call form is:

```cpp
const auto generate = rlf::sdk::make_pipeline(
    rlf::sdk::PipelineTask::text_generation,
    "/models/solstice-preview"
);
```

After installation, downstream CMake projects can use:

```cmake
find_package(RLF 11 CONFIG REQUIRED)
target_link_libraries(my_application PRIVATE rlf::core)
```

Multimodal requests use the same interface:

```cpp
const rlf::sdk::Pipeline vision(
    rlf::sdk::PipelineTask::image_text_to_text,
    model
);
const auto answer = vision({
    .prompt = "Describe this image.",
    .image = "/data/example.png",
});
```

Tool execution is opt-in and uses the existing typed, policy-controlled runtime:

```cpp
rlf::sdk::PipelineOptions options;
options.register_safe_tools = true;
options.tool_policy.allow_file_reads = false;

const rlf::sdk::Pipeline tools(
    rlf::sdk::PipelineTask::tool_use,
    model,
    options
);
```

## Training API

`Trainer` exposes RLF-native typed records rather than gradient batches:

```cpp
#include <rlf/sdk/trainer.hpp>

rlf::sdk::Trainer trainer(rlf::sdk::AutoModel::from_profile(
    rlf::solstice::SolsticeProfile::preview_6g
));

trainer.train(rlf::sdk::DialogueTrainingRecord{
    .prompt = "What is the codename?",
    .response = "The codename is Aurora.",
});

trainer.save_pretrained(
    "/models/aurora",
    {.name = "aurora", .profile = rlf::solstice::SolsticeProfile::preview_6g}
);
```

Supported record types are text corpora, dialogues, images, instructions, preferences, tool routes,
facts, and relational rules. This convenience API is suitable for local experiments. Large or
publishable training runs should continue using the audited immutable-ledger workflow so provenance,
deduplication, contamination checks, and shard resume remain enforced.

## Design boundary

This SDK is intentionally local-first. `from_pretrained` accepts a checkpoint file or bundle
directory, never downloads code or model state, and never enables tools implicitly. A future
registry can resolve names to downloaded immutable bundles without changing the loading or pipeline
interfaces.
