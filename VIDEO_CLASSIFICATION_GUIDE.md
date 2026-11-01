# Video Classification with Gemini 3 Flash

## Overview

Your LLM photo classification script now supports **video classification** using Gemini 3 Flash's video understanding capabilities!

## How It Works

- **Same model**: Uses `gemini-3.8-flash` (same as for photos)
- **Files API**: Videos are uploaded to Gemini, processed, then analyzed
- **Same categories**: Share, Storage, or Ignore
- **Video-specific**: Also evaluates audio quality and video quality

## Usage

### Default Behavior (No Classification)

By default, videos are simply copied to the `Videos/` folder without classification:

```bash
python taste_classify_gemini_v4.py "Camera Roll\2025"
```

Output:
```
📹 Found 589 videos - copying to Videos/ folder
   (Use --classify-videos to classify videos with Gemini)
   ✅ Videos: 589 copied, 0 already exist
```

### Enable Video Classification

Add the `--classify-videos` flag to classify videos:

```bash
python taste_classify_gemini_v4.py "Camera Roll\2025" --classify-videos
```

## Cost Estimation - UPDATED WITH ACTUAL PRICING

The script automatically estimates cost **before** processing:

```
📹 Found 1415 videos - CLASSIFICATION MODE ENABLED
   Video stats:
      Count: 1415 videos
      Total size: 80.3 GB
      Estimated duration: ~1416 minutes (60 sec/video avg)

   Cost breakdown:
      Input tokens: ~2,244,750 (video frames + prompts)
      Output tokens: ~424,500 (JSON responses)
      Input cost: $0.67
      Output cost: $1.06
      TOTAL COST: $1.73

   ⚠️  This is 1415 videos × $0.0012/video
   ⚠️  Estimate assumes ~60 sec/video based on file size
   ⚠️  Actual cost may be 2-3x higher if videos are longer than estimated

   💡 RECOMMENDATION: Test with 10-20 videos first to verify accuracy
```

### Gemini 3.8 Flash Pricing (Oct 2026)

From https://ai.google.dev/gemini-api/docs/pricing:

- **Input tokens**: $0.75 per 1M tokens (text/image/video)
- **Output tokens**: $3.75 per 1M tokens (JSON responses)

### Cost Breakdown Per Video

**For a 60-second video:**
- Input: 60 sec × 300 tokens/sec = 18,000 visual tokens
- Input: +1,500 tokens (rich taste profile prompt)
- Output: +300 tokens (JSON with reasoning)
- **Input cost**: 19,500 × $0.75 / 1M = $0.0146
- **Output cost**: 300 × $3.75 / 1M = $0.0011
- **Total: ~$0.016 per minute of video**

### Real-World Examples

| Video Count | Avg Duration | Total Size | Estimated Cost | Notes |
|------------|-------------|-----------|----------------|-------|
| 10 videos | 30 sec | 150 MB | $0.06 | Good for testing |
| 100 videos | 60 sec | 1.5 GB | $1.10 | Small batch |
| 589 videos | 45 sec | 9 GB | $4.86 | Medium batch |
| 1,415 videos | 60 sec | 20 GB | $15.57 | Large batch |
| **5,000 videos** | **60 sec** | **80 GB** | **$55.00** | **Your full library** |

### Important Notes

⚠️ **File size ≠ duration**: The script estimates duration from file size (rough), but actual duration may be 2-3x different

⚠️ **Output costs matter**: Output tokens cost 6x more than input ($3.00 vs $0.50 per million)

⚠️ **Test first**: Run on 10-20 videos, check actual cost in Google AI Studio, then extrapolate

**Recommendation**: For 80 GB of video, budget $50-100 depending on average video length.

## Output

### During Classification

```
   Video 1/589: 20220815_video.mp4
      Uploading video to Gemini...
      Video processed, analyzing...
      Share   → Share   (0.92) | Audio: good, Video: good

   Video 2/589: 20220816_tantrum.mp4
      Uploading video to Gemini...
      Video processed, analyzing...
      Storage → Storage (0.45) | Audio: poor, Video: good
```

### Video Routing

Classified videos are routed to folders just like photos:

- **Share/** - Share-worthy videos (special moments, good quality)
- **Storage/** - Videos of children but not share-worthy
- **Ignore/** - No children visible or inappropriate content
- **Review/** - Uncertain classifications

## Evaluation Criteria

Videos are evaluated on:

1. **Contains children?** - Must show young children
2. **Appropriate?** - Family-friendly, no private parts
3. **Audio quality** - Joyful sounds vs. constant crying/screaming
4. **Video quality** - Stable, clear vs. shaky, dark
5. **Special moment?** - Personality, milestone, genuine interaction

## Video-Specific Prompt

The prompt is adapted for videos:

```
**SHARE = Videos worth sharing with family**
- Captures child's personality, special moment, or milestone
- Good audio quality (joyful sounds, laughter, NOT constant crying/screaming)
- Reasonable video quality (not too shaky, decent lighting)
- Parent interactions are positive and engaging

**STORAGE = Videos of children that aren't share-worthy**
- Poor quality (too shaky, bad lighting, can't see faces)
- Excessive crying/screaming/tantrum throughout
- Boring/uneventful (nothing special happening)
- Too long with no clear focus
```

## Caching

- Video classifications are **cached** just like photos
- Cache key: `video_{hash_of_file_path_size_mtime}`
- Re-running won't re-process already classified videos
- Cache stored in `.taste_cache/labeling_pairwise/gemini_cache_v4_grouped_{folder}.json`

## Limitations

1. **One video per request** (can't batch like photos)
2. **Processing time**: Each video takes ~5-10 seconds to upload + process
3. **Max duration**: Up to 2 hours per video (at default resolution)
4. **Cost**: Much more expensive than photos (~300 tokens/second)

## Testing Workflow

### Step 1: Test with a Small Batch

Create a test folder with 5-10 videos:

```bash
# Copy a few videos to test folder
mkdir "E:\test_videos"
# Copy 5-10 representative videos

# Run classification
python taste_classify_gemini_v4.py "E:\test_videos" --classify-videos
```

### Step 2: Review Results

Check the output folders:
- Are videos routed correctly?
- Are audio/video quality assessments accurate?
- Are special moments being identified?

### Step 3: Full Batch

Once satisfied, run on full folders:

```bash
python taste_classify_gemini_v4.py "Camera Roll\2025" --classify-videos
```

## Examples

### Share-Worthy Video

```json
{
    "classification": "Share",
    "confidence": 0.92,
    "reasoning": "Toddler's first steps with joyful reactions from parents",
    "contains_children": true,
    "is_appropriate": true,
    "audio_quality": "good",
    "video_quality": "good"
}
```

### Storage Video

```json
{
    "classification": "Storage",
    "confidence": 0.45,
    "reasoning": "Baby crying throughout, video is very shaky and dark",
    "contains_children": true,
    "is_appropriate": true,
    "audio_quality": "poor",
    "video_quality": "poor"
}
```

### Ignore Video

```json
{
    "classification": "Ignore",
    "confidence": 1.0,
    "reasoning": "Video of food preparation only, no people visible",
    "contains_children": false,
    "is_appropriate": true,
    "audio_quality": "n/a",
    "video_quality": "good"
}
```

## Tips

1. **Start small** - Test with 5-10 videos first
2. **Check costs** - Look at the estimate before proceeding
3. **Review quality** - Check if classifications match your expectations
4. **Adjust as needed** - Videos with lots of crying might be classified as Storage
5. **Be patient** - Processing is slower than photos (5-10 seconds per video)

## Comparison: With vs. Without Classification

### Without `--classify-videos` (Default)

```
📹 Found 589 videos - copying to Videos/ folder
   ✅ Videos: 589 copied, 0 already exist
```

**Result**: All 589 videos in `Videos/` folder (unclassified)

### With `--classify-videos`

```
📹 Found 589 videos - CLASSIFICATION MODE ENABLED
   Estimated cost: $0.025

   Video 1/589: ...
   ...
   ✅ 589 videos classified
```

**Result**: Videos routed to Share/Storage/Ignore folders based on quality

## Questions?

- **"Is it worth it?"** - Depends on your video library. If you have hundreds of videos and want to find the share-worthy ones automatically, yes!
- **"How accurate is it?"** - Test with a small batch first. Gemini can identify special moments, audio quality, and technical quality.
- **"Can I re-process?"** - Yes! Results are cached, so re-running is fast and free (already processed videos are skipped).
