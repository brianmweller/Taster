<p align="center">
  <img src="https://raw.githubusercontent.com/brianmweller/Taster/master/assets/Readme_Logo.png" alt="Taster" width="200">
</p>

<h1 align="center">Taster</h1>

<p align="center"><strong>Teach AI your taste. Apply it to everything.</strong></p>

You know quality when you see it -- but writing down *why* is the hard part. Taster figures it out for you. Show it examples you like and it reverse-engineers your standards into a reusable profile: a human-readable style guide that doubles as an executable classifier.

Sort 1,000 family photos in 10 minutes for $1.30. Extract coding standards from your best files. Grade 200 essays against criteria you never had to write by hand.

---

## Demo

**280 photos + videos classified in 12 minutes for $0.30** (10 parallel workers, Gemini Flash).

<p align="center">
  <img src="https://raw.githubusercontent.com/brianmweller/Taster/master/assets/demo_terminal.png" alt="Terminal output showing classification of 280 files" width="780">
</p>

Every file gets a score, a plain-English reason, and per-dimension diagnostic scores -- all grounded in the taste profile's criteria:

<p align="center">
  <img src="https://raw.githubusercontent.com/brianmweller/Taster/master/assets/demo_reasoning.png" alt="AI reasoning for each classification decision" width="780">
</p>

Burst photos taken seconds apart are compared head-to-head -- AI picks the best and explains why:

<p align="center">
  <img src="https://raw.githubusercontent.com/brianmweller/Taster/master/assets/demo_burst.png" alt="Burst intelligence: AI picks the best from a series" width="780">
</p>

Dimension scores show *why* each file scored the way it did -- which priorities drove the decision:

```
8_hero_share.jpg [Share] score=5
  parent_child_interaction_quality:        4
  baby_expression_mischief_joy_engagement: 5
  parent_expressions_should_be_engaged:    4
  baby_face_clearly_visible:               5
  genuine_emotional_moments:               5

4_technical_pass_share.jpg [Storage] score=2
  parent_child_interaction_quality:        1  ← no parent in frame
  baby_expression_mischief_joy_engagement: 2
  parent_expressions_should_be_engaged:    1  ← no parent in frame
  baby_face_clearly_visible:               5
  genuine_emotional_moments:               2
```

Dimensions are auto-derived from the profile's `top_priorities`. When a photo scores high on face visibility but low on interaction and emotion, you know *exactly* what to adjust.

Full score distribution and category breakdown:

<p align="center">
  <img src="https://raw.githubusercontent.com/brianmweller/Taster/master/assets/demo_stats.png" alt="Score distribution and category breakdown" width="780">
</p>

---

## How It Works

1. **Classify** -- AI evaluates each file against your profile's criteria, sorts it into categories, and returns per-dimension diagnostic scores
2. **Learn** -- Feed results back in with corrections, and Taster uses dimension scores to pinpoint what needs adjusting
3. **Apply** -- The refined profile becomes a sharper classifier for future batches

Each cycle sharpens the profile. What starts as "sort my photos" becomes a rich, nuanced document that captures exactly how you think about quality -- then enforces it automatically.

---

## What People Use It For

### Family Photos

The original use case. Ships with a built-in profile that classifies photos into Share / Storage / Review / Ignore. Run it on a year of camera roll and get the 200 worth sharing for about a dollar.

<details>
<summary><strong>Example: AI-generated taste profile from a family photo library</strong></summary>

Nobody wrote this by hand -- it was synthesized from examples:

> **Philosophy:** Photos worth sharing capture genuine moments of connection and joy between baby and loved ones. They should highlight the baby's expressions, engagement with the world, and the loving interactions they share, all presented in a visually clear and appealing manner.
>
> **Top priorities (ranked):**
> 1. Baby's face is visible, well-lit, and in focus
> 2. Baby actively engaging with people or objects
> 3. Clear and expressive facial expressions (joy, curiosity, mischief)
> 4. Positive interactions between baby and parent/caregiver
> 5. Well-framed photos that focus on the baby and minimize distractions
> 6. Photos that evoke positive emotions -- joy, love, wonder, humor
> 7. Good lighting, sharpness, and minimal blur
>
> **Must-have:** Baby's face clearly visible and in focus. Baby engaged in the activity or with another person. Well-lit and sharp. Expressive and genuine emotion conveyed.
>
> **Highly valued:** Interactions between baby and parent (especially eye contact or physical touch). Baby displaying positive emotions. Photos that tell a story or capture a special moment.
>
> **Deal-breakers:** Baby's face not visible or obscured. Blurry or out of focus. Poor lighting. Baby displaying negative expression without context. Child too small in frame.
>
> **Burst guidance:** Prioritize the clearest expressions, best focus on the baby's face, and minimal distractions. Choose the photo that best captures the moment's essence.
>
> **Specific guidance:**
> - *"Does this photo capture the baby's personality and current stage of development?"*
> - *"If both parent and baby are in the photo, ensure both have engaged and positive expressions."*
> - *"Reject photos even if technically good if the baby looks unhappy (unless comically so)."*

That profile is both a human-readable style guide *and* an executable classifier. Every future photo is evaluated against those exact criteria.

</details>

### Grading at Scale

Grade 200 student submissions against *your* standards. Create a profile from a handful of A-grade and C-grade examples. Taster extracts what your grading decisions had in common and applies them to the full batch -- each result includes a brief explanation of why the submission landed where it did, grounded in the profile's criteria.

### Code Standards

Start with a code quality profile -- generate one from a quick description or write your own. Taster classifies every file into tiers (Exemplary / Solid / Needs Work) with explanations. Review the results, correct what you disagree with, and refine the profile. Each cycle sharpens it until the profile captures the standards you actually follow -- error handling patterns, naming conventions, architectural choices. That profile becomes both a style guide and an automated reviewer for future code.

### Multi-File Packages

Some things can't be evaluated one file at a time -- a vendor sends an MSA, a security questionnaire, and a product deck; a research submission includes a paper, dataset description, and methodology appendix; a creative portfolio has case studies, a process document, and references. `--bundles` mode treats each subfolder as a single unit. The AI sees every file in the package together and returns one holistic classification -- exactly how a human reviewer works.

### Any Collection

Research papers, product photos, legal documents, design assets. Anything where you "know it when you see it" but can't write the rules upfront. Taster closes the gap between your tacit taste and explicit, repeatable judgment.

---

## Taste Profiles

Profiles define *what* you're classifying, *how* you want it categorized, and *what criteria* to use.

### Example Profile

```json
{
  "name": "student-essays",
  "categories": [
    {"name": "Strong", "description": "Excellent work, minimal feedback needed"},
    {"name": "Developing", "description": "Shows promise, needs specific feedback"},
    {"name": "Needs Work", "description": "Significant gaps to address"}
  ],
  "top_priorities": ["Clear thesis supported by evidence", "Logical structure", "Engagement with sources"],
  "philosophy": "Reward critical thinking and clear communication over polish.",
  "thresholds": {"Strong": 0.70, "Developing": 0.40}
}
```

Profiles are stored as JSON in `profiles/`. Taster ships with starter profiles for common use cases -- family photos, code review, student essays, and product photography. See the full schema in `profiles/default-photos.json`.

### Ways to Create a Profile

| Method | Input | Time | Best for |
|--------|-------|------|----------|
| Quick Profile | Plain English description | ~10s | Getting started fast |
| Generate from Examples | Example folders per category | ~2 min | When you have sorted examples |
| **Pairwise Training** | 15-50+ side-by-side comparisons | 15-30 min | **Highest accuracy** |

- **Quick profile:** Ask Claude *"Create a profile for sorting research papers into Keep, Skim, and Skip"* and it generates one from your description. Great for a first pass.
- **From examples:** Point `taster_generate_profile` at a folder of good examples and a folder of bad examples (binary mode), or provide `category_folders` mapping category names to folders for 3+ categories (e.g. `{"Excellent": "path", "Mediocre": "path", "Poor": "path"}`). Taster analyzes each category and synthesizes criteria.
- **Pairwise training:** Launch the Gradio trainer (`taster train <folder>`) for the highest-fidelity option. Compare photos side-by-side, pick keepers from burst galleries, and synthesize a profile that captures exactly how you think about quality.
- **By hand:** Write a JSON file directly in `profiles/`.

### Training and Refinement

Profiles improve over time:

1. **Pairwise training** -- Run `taster train <folder>` to launch a Gradio UI. Compare photos side-by-side, pick keepers from burst galleries, and synthesize a profile when you have enough labels (15+).

2. **Corrective refinement** -- After classifying a folder, correct the results you disagree with and call `taster_refine_profile` in Claude Desktop. Taster analyzes the gap between its predictions and your corrections -- including per-dimension scores that reveal *which* criteria are miscalibrated -- and adjusts priorities, thresholds, and guidance. Repeat each batch to continuously sharpen the profile.

3. **Simple feedback** -- Submit individual corrections via `taster_submit_feedback` for lightweight feedback without full refinement.

---

## Getting Started

Taster runs as an MCP server inside Claude Desktop, or as a standalone CLI / REST API.

**Prerequisites:** Python 3.12+, at least one AI provider API key.

### Install

```bash
pip install taster[gemini]        # Gemini (recommended -- cheapest)
# or: pip install taster[openai]  # OpenAI
# or: pip install taster[all]     # Everything
```

<details>
<summary><strong>Install from source</strong></summary>

```bash
git clone https://github.com/brianmweller/Taster.git
cd Taster
pip install -e ".[gemini]"
```

</details>

### First-run setup

```bash
taster init
```

Creates your config directory, prompts for API keys, and optionally wires up Claude Desktop. Config files are stored in a platform-appropriate location:
- **Windows:** `%APPDATA%\taster\`
- **macOS:** `~/Library/Application Support/taster/`
- **Linux:** `$XDG_CONFIG_HOME/taster/` (default `~/.config/taster/`)

### Claude Desktop (Recommended)

Connect Taster to Claude Desktop and use it conversationally. No command-line needed after setup. Taster's MCP server is compatible with any MCP host -- the examples below show Claude Desktop, but it works with any app that supports the [Model Context Protocol](https://modelcontextprotocol.io).

If `taster init` configured Claude Desktop for you, just restart Claude Desktop. Otherwise, add to your `claude_desktop_config.json`:

```json
{
  "mcpServers": {
    "taster": {
      "command": "taster",
      "args": ["serve"],
      "env": {
        "PYTHONIOENCODING": "utf-8",
        "PYTHONUTF8": "1"
      }
    }
  }
}
```

<details>
<summary><strong>Claude Desktop config locations</strong></summary>

- **Windows:** `%APPDATA%\Claude\claude_desktop_config.json`
- **macOS:** `~/Library/Application Support/Claude/claude_desktop_config.json`
- **Linux:** `$XDG_CONFIG_HOME/Claude/claude_desktop_config.json`

</details>

<details>
<summary><strong>Running from a cloned repo (alternative)</strong></summary>

```json
{
  "mcpServers": {
    "taster": {
      "command": "python",
      "args": ["mcp_server.py"],
      "env": {
        "PYTHONPATH": "/path/to/Taster",
        "PYTHONIOENCODING": "utf-8",
        "PYTHONUTF8": "1"
      }
    }
  }
}
```

</details>

Restart Claude Desktop. Ask it: *"Check my Taster status"* to verify everything is connected. Then just talk to it:
- *"Sort the photos in my Camera Roll folder"*
- *"Create a profile for grading student essays"*
- *"Generate a taste profile from my best code examples in src/"*
- *"Refine my photo profile -- I disagreed with some of the results"*

<details>
<summary><strong>All MCP tools</strong> (13 tools)</summary>

| Tool | What it does |
|------|-------------|
| `taster_status` | Check setup status (API keys, providers, profiles) |
| `taster_list_profiles` | List all taste profiles |
| `taster_get_profile` | Get profile details with human-readable summary |
| `taster_create_profile` | Create a new profile with full control |
| `taster_update_profile` | Update specific fields of an existing profile |
| `taster_delete_profile` | Delete a profile (with confirmation) |
| `taster_quick_profile` | AI-generate a profile from a plain English description |
| `taster_generate_profile` | AI-generate a profile from example files (binary good/bad or multi-category) |
| `taster_classify_folder` | Classify all files in a folder |
| `taster_classify_files` | Classify specific files by path |
| `taster_submit_feedback` | Submit classification corrections |
| `taster_view_feedback` | Review all feedback and stats |
| `taster_refine_profile` | Refine profile from classification corrections (AI) |

Pairwise training (side-by-side photo comparison and profile synthesis) is handled by the standalone Gradio trainer: `taster train <folder>`.

</details>

<details>
<summary><strong>CLI</strong></summary>

```bash
# Classify a folder (auto-detects media types and provider)
taster classify ~/Photos/MyFolder

# Use a specific taste profile
taster classify ~/Photos/MyFolder --profile default-photos

# Force a specific AI provider
taster classify ~/Photos/MyFolder --provider openai

# Dry run (test without moving files)
taster classify ~/Photos/MyFolder --dry-run

# Bundle mode (one classification per subfolder)
taster classify ~/Submissions --bundles --profile vendor-review

# Check setup status
taster status
```

Files are sorted into `MyFolder_sorted/` with subfolders matching your profile's categories.

</details>

<details>
<summary><strong>REST API</strong></summary>

```bash
python serve.py                          # Start on localhost:8000
python serve.py --host 0.0.0.0 --port 9000  # Custom host/port
python serve.py --reload                 # Development mode
```

Interactive docs at `http://localhost:8000/docs`.

| Method | Endpoint | Description |
|--------|----------|-------------|
| GET | `/api/profiles/` | List all profiles |
| GET | `/api/profiles/{name}` | Get profile details |
| POST | `/api/profiles/` | Create new profile |
| PUT | `/api/profiles/{name}` | Update profile |
| DELETE | `/api/profiles/{name}` | Delete profile |
| POST | `/api/classify/folder` | Classify a local folder |
| GET | `/api/classify/{job_id}` | Get job status |
| GET | `/api/classify/{job_id}/results` | Get results |
| GET | `/api/results/{job_id}/export` | Export as CSV |
| POST | `/api/training/feedback` | Submit feedback |
| GET | `/api/training/stats` | Get training stats |

</details>

---

## AI Providers

Taster supports three cloud AI providers and any local LLM with an OpenAI-compatible endpoint. Install only the SDK(s) you need.

| Feature | Gemini | OpenAI (GPT-5) | Anthropic (Claude) | Local LLM |
|---------|--------|---------------------|-------------------|-----------|
| Images | Native | Base64 | Base64 | Base64 |
| Videos | Native upload | Frame extraction | Frame extraction | Frame extraction |
| Audio | Native upload | Text fallback | Text fallback | Text fallback |
| PDFs | Native upload | Page-to-image | Native | Page-to-image |
| Relative cost | Cheapest | Mid | Most expensive | Free (your hardware) |
| Env var | `GEMINI_API_KEY` | `OPENAI_API_KEY` | `ANTHROPIC_API_KEY` | `LOCAL_LLM_URL` |

Gemini is the recommended default -- cheapest and has native video/PDF support. The system auto-detects which key is available (gemini > openai > anthropic > local), or you can force a provider with `--provider` or in `config.yaml`.

### Local LLM Setup

Run classification entirely on your machine -- no data leaves your network. Taster connects to any server that speaks the OpenAI chat-completions API.

| Server | Install | Default URL |
|--------|---------|-------------|
| [Ollama](https://ollama.com) | `ollama serve` then `ollama pull llama3.2` | `http://localhost:11434/v1` |
| [LM Studio](https://lmstudio.ai) | Download, load a model, start server | `http://localhost:1234/v1` |
| [vLLM](https://docs.vllm.ai) | `pip install vllm && vllm serve llama3.2` | `http://localhost:8000/v1` |

**Quick start (Ollama):**

```bash
ollama pull llama3.2               # Download model (~2 GB)
export LOCAL_LLM_URL=http://localhost:11434/v1
taster classify ~/Photos --provider local
```

Or add to `config.yaml`:

```yaml
model:
  provider: local
  local_model: llama3.2
  local_base_url: http://localhost:11434/v1
```

Local models are slower than cloud APIs and may produce lower-quality classifications for complex media. Best for text/document classification or when privacy is the top priority. Combine with `--bundles` to review multi-file packages (vendor packets, portfolios, submissions) entirely on-device.

---

## Cost Estimates

Gemini 3 Flash pricing (cheapest provider):

| Item | Cost |
|------|------|
| Photo | ~$0.0013 |
| Burst photo | ~$0.00043 (shared prompt) |
| Video | ~$0.011 per minute |
| Audio | ~$0.006 per minute |
| Document | ~$0.002-0.01 (varies by size) |

**Example:** 1,000 photos = ~$1.30 (first run), ~$0 (cached re-run).

OpenAI and Anthropic cost more per call. See provider pricing pages for current rates.

---

## Configuration

All settings are in `config.yaml`. Key sections:

```yaml
model:
  provider: null          # null = auto-detect. Options: gemini, openai, anthropic, local
  name: "gemini-3-flash-preview"
  openai_model: "gpt-5.2"
  anthropic_model: "claude-sonnet-4-6"
  local_model: "llama3.2"
  local_base_url: "http://localhost:11434/v1"

classification:
  share_threshold: 0.60       # Confidence threshold for top category
  review_threshold: 0.35      # Below this → lowest category
  classify_videos: true
  parallel_photo_workers: 10  # Concurrent photo classification workers
  parallel_video_workers: 10  # Concurrent video classification workers
  classify_audio: true
  parallel_audio_workers: 10  # Concurrent audio classification workers

profiles:
  profiles_dir: "profiles"
  active_profile: "default-photos"
  auto_detect_media_type: true

burst_detection:
  time_window_seconds: 10
  embedding_similarity_threshold: 0.92

document:
  max_file_size_mb: 50.0
  max_pages_per_document: 20
  enable_text_embeddings: true
  similarity_threshold: 0.85
```

See `config.yaml` for all options with inline documentation.

---

## Supported Formats

| Type | Extensions |
|------|-----------|
| **Images** | `.jpg` `.jpeg` `.png` `.gif` `.webp` `.heic` `.tif` `.tiff` `.bmp` |
| **Videos** | `.mp4` `.mov` `.avi` `.mkv` `.m4v` `.3gp` `.wmv` `.flv` `.webm` |
| **Audio** | `.mp3` `.wav` `.flac` `.aac` `.ogg` `.m4a` `.wma` `.opus` `.aiff` |
| **Documents** | `.pdf` `.docx` `.xlsx` `.pptx` `.html` `.htm` `.txt` `.md` `.csv` `.rtf` |
| **Source Code** | `.py` `.js` `.ts` `.jsx` `.tsx` `.java` `.go` `.rs` `.rb` `.cpp` `.c` `.cs` `.swift` `.kt` `.php` `.lua` `.scala` + more |
| **Config / IaC** | `.yaml` `.yml` `.toml` `.json` `.xml` `.tf` `.hcl` `.dockerfile` |
| **Shell** | `.sh` `.bash` `.zsh` `.ps1` `.sql` |

PDFs, videos, and audio are uploaded natively to Gemini. For OpenAI and Anthropic, videos are frame-extracted and PDFs are rendered to images automatically. Audio requires Gemini for full analysis.

---

## Output Structure

```
MyFolder_sorted/
├── Share/                       # Share-worthy photos/videos
├── Storage/                     # Keep but don't share
├── Review/                      # Needs manual review
├── Ignore/                      # Not relevant
└── Reports/                     # Classification logs + summary
```

For custom profiles, output folders match the profile's category names (e.g., `Strong/`, `Maybe/`, `Pass/`).

---

## Reference

<details>
<summary><strong>Error Handling</strong></summary>

Automatic retry with exponential backoff for API failures.

| Error Type | Retriable | Description |
|------------|-----------|-------------|
| `api_error` | Yes | General API failures |
| `timeout` | Yes | Request timeout |
| `rate_limit` | Yes | API rate limiting |
| `invalid_response` | Yes | Malformed API response |
| `safety_blocked` | No | Content blocked by safety filters |
| `load_error` | No | Failed to load file |

Reprocess failures:
```bash
python clear_failed_for_retry.py "path/to/sorted" --dry-run  # Preview
python clear_failed_for_retry.py "path/to/sorted"            # Clear and re-run
```

</details>

<details>
<summary><strong>Architecture</strong></summary>

```
taster/
├── core/                  # Configuration, caching, AI clients, profiles
│   ├── ai_client.py       # AIClient ABC + AIResponse
│   ├── provider_factory.py # Auto-detect provider from API keys
│   ├── media_prep.py      # Video frame extraction, PDF rendering, base64 encoding
│   └── providers/         # Gemini, OpenAI, Anthropic implementations
├── features/              # Quality scoring, burst detection, embeddings
├── classification/        # Prompt building, AI classification, confidence routing
├── pipelines/             # Photo, document, and mixed orchestration
├── api/                   # REST API (FastAPI)
├── training/              # Pairwise training & profile synthesis
└── mcp/                   # MCP server (Claude Desktop)
```

| Command | Purpose |
|---------|---------|
| `taster classify <folder>` | Classify files against a profile |
| `taster train <folder>` | Launch Gradio pairwise trainer |
| `taster serve` | Start MCP server for Claude Desktop |
| `taster init` | Interactive first-run setup |
| `taster status` | Show config, profiles, API key status |

</details>

<details>
<summary><strong>Development</strong></summary>

```bash
pip install -e ".[gemini,dev]"               # Install for development
pytest tests/ -v                              # Run tests
pytest tests/ --cov=taster --cov-report=html     # With coverage
```

</details>

<details>
<summary><strong>Troubleshooting</strong></summary>

**"No AI provider configured"** -- Create `.env` with at least one API key (`GEMINI_API_KEY`, `OPENAI_API_KEY`, or `ANTHROPIC_API_KEY`).

**"No module named 'taster'"** -- Install with `pip install -e .` or run from the project root directory.

**Low share rate (<20%)** -- Lower `classification.share_threshold` in `config.yaml` (e.g., 0.50).

**Document extraction issues** -- Install document support: `pip install taster[documents]`.

</details>

---

Built with Google Gemini, OpenAI, Anthropic, OpenCLIP, sentence-transformers, FastAPI, MCP SDK, Gradio, and Claude Code.

**Version:** 3.6.8 | **PyPI:** [`taster`](https://pypi.org/project/taster/) | **Last Updated:** April 2026

<p align="center">Built by <a href="https://github.com/brianmweller">@brianmweller</a></p>
