# Windows 6 GB Conversation Demo

This workflow trains a small RLF checkpoint from blank on locally generated conversation data. It
does not download or scrape web text, so there is no unknown-license training content. The generated
dataset is marked `CC0-1.0` and contains narrow identity, RLF, context, safety, coding, arithmetic,
instruction, preference, and fact examples.

## Requirements

- Windows 10 or 11.
- Visual Studio 2026 or 2022 with **Desktop development with C++**.
- CMake 4.2 or newer for Visual Studio 2026; CMake 3.25 or newer for Visual Studio 2022.
- For GPU execution: NVIDIA CUDA Toolkit compatible with the installed driver.
- Approximately 12 GiB available system RAM is the profile estimate. The profile GPU working-set
  estimate is 5 GiB, so a 6 GB card is suitable for this small test but has little headroom.

## Train

Open PowerShell in the repository:

```powershell
.\TRAIN_PREVIEW_6GB_WINDOWS.bat `
  -Backend cuda `
  -AssistantName Aurora `
  -OwnerName "Your Name"
```

The script:

1. Detects Visual Studio 2026 or 2022 and builds the matching CUDA preset if needed.
2. Generates data under `demo_data\preview_conversation`.
3. Creates a blank `preview-6g` model and trains its tokenizer/text state.
4. Trains dialogue, instruction, preference, and fact records.
5. Saves and verifies `models\preview_conversation_6gb.rlfsp`.

To deliberately replace an existing demo checkpoint:

```powershell
.\TRAIN_PREVIEW_6GB_WINDOWS.bat -Backend cuda -Reset
```

If CUDA compilation or execution is unavailable, the same workflow works on CPU:

```powershell
.\TRAIN_PREVIEW_6GB_WINDOWS.bat -Backend optimized_cpu
```

## Converse

```powershell
.\CHAT_PREVIEW_6GB_WINDOWS.bat -Backend cuda
```

This starts the stateful no-tools chat path with a 256-token conversation budget, 12 retained turns,
and responses capped at 96 tokens.

Commands:

- `/context` displays current token and eviction statistics.
- `/clear` removes conversation history.
- `/image PATH` attaches an image.
- `/clear-image` removes the image.
- `/quit` exits.

Adjust the chat window if needed:

```powershell
.\CHAT_PREVIEW_6GB_WINDOWS.bat `
  -Backend cuda `
  -ContextTokens 192 `
  -ChatTurns 8 `
  -MaximumResponseTokens 64
```

## Adding personal conversation data

Append tab-separated rows to:

```text
demo_data\preview_conversation\dialogues.tsv
```

Each row is:

```text
prompt<TAB>response<TAB>optional grounding
```

Then retrain with `-Reset`. Keep each field on one line and do not place tabs inside a field.
For experiments sourced from the internet, use only material whose license permits model training,
record its exact source and license, and move to the repository's audited ledger workflow before
making publishable claims.
