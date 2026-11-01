#!/usr/bin/env python3
"""
Generate rich taste profile from training labels + visual analysis.

This script:
1. Loads all pairwise training labels
2. Reads actual Share and Storage photos
3. Uses Gemini to analyze:
   - Visual patterns in Share photos (what makes them share-worthy?)
   - Visual patterns in Storage photos (what makes them rejectable?)
   - User's reasoning text patterns
4. Generates comprehensive taste_preferences.json

Usage:
    python generate_taste_profile.py
"""

import os
import sys
import json
from pathlib import Path
from collections import defaultdict
import google.generativeai as genai
from PIL import Image, ImageOps

# Add project to path
sys.path.insert(0, str(Path(__file__).parent))

from taste_sort_win import CACHE_ROOT
from taste_classify_gemini_v4 import (
    LABELS_FILE, convert_pairwise_labels_to_training_examples,
    fix_image_orientation
)

# Load .env
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "")
if GEMINI_API_KEY:
    genai.configure(api_key=GEMINI_API_KEY)

OUTPUT_FILE = Path("taste_preferences_generated.json")

def load_training_data():
    """Load and process training labels."""
    print("📚 Loading training data...")

    if not LABELS_FILE.exists():
        print(f"❌ No training labels found at {LABELS_FILE}")
        return None

    with open(LABELS_FILE, 'r') as f:
        labels = json.load(f)

    training_examples = convert_pairwise_labels_to_training_examples(labels)

    # Separate into Share and Storage
    share_examples = {path: data for path, data in training_examples.items()
                     if data["action"] == "Share"}
    storage_examples = {path: data for path, data in training_examples.items()
                       if data["action"] == "Storage"}

    print(f"   ✅ Loaded {len(training_examples)} training examples")
    print(f"      Share: {len(share_examples)}")
    print(f"      Storage: {len(storage_examples)}")

    return share_examples, storage_examples

def analyze_reasoning_text(share_examples, storage_examples):
    """Extract patterns from user's reasoning text."""
    print("\n🔍 Analyzing reasoning text patterns...")

    share_reasons = [data.get("reasoning", "") for data in share_examples.values()]
    storage_reasons = [data.get("reasoning", "") for data in storage_examples.values()]

    # Combine all reasoning
    share_text = "\n".join([f"- {r}" for r in share_reasons if r])
    storage_text = "\n".join([f"- {r}" for r in storage_reasons if r])

    prompt = f"""Analyze this user's photo sorting reasoning and extract their taste preferences.

**SHARE PHOTOS - User's reasoning for photos they SHARE:**
{share_text}

**STORAGE PHOTOS - User's reasoning for photos they REJECT:**
{storage_text}

Analyze the patterns and respond with JSON:
{{
    "valued_qualities": [
        "list of qualities, themes, and attributes the user values",
        "e.g., parent-child interaction, joyful expressions, engagement"
    ],
    "reject_criteria": [
        "list of qualities, themes, and attributes the user rejects",
        "e.g., can't see face, vacant expressions, poor framing"
    ],
    "specific_preferences": [
        "specific, concrete preferences extracted from reasoning",
        "e.g., both subjects should be engaged, not looking different directions"
    ],
    "emotional_keywords": [
        "emotional qualities the user values",
        "e.g., mischief, joy, intimacy, engagement"
    ],
    "technical_priorities": [
        "technical aspects the user prioritizes",
        "e.g., sharpness, lighting, visible faces, framing"
    ],
    "contextual_insights": [
        "higher-level insights about user's taste",
        "e.g., values interaction over solo shots, parent expressions matter as much as baby"
    ]
}}

Extract ONLY what's clearly stated or strongly implied. Be specific and concrete."""

    if not GEMINI_API_KEY:
        print("   ⚠️  No Gemini API key, skipping reasoning analysis")
        return None

    try:
        model = genai.GenerativeModel('gemini-3.8-flash')
        response = model.generate_content(prompt)
        text = response.text.strip()

        # Parse JSON
        import re
        json_match = re.search(r'\{.*\}', text, re.DOTALL)
        if json_match:
            result = json.loads(json_match.group(0))
            print(f"   ✅ Extracted reasoning patterns")
            return result
        else:
            print(f"   ⚠️  Could not parse reasoning analysis")
            return None

    except Exception as e:
        print(f"   ⚠️  Error analyzing reasoning: {e}")
        return None

def analyze_share_photos_visual(share_examples, max_photos=15):
    """Use Gemini to analyze visual patterns in Share photos."""
    print(f"\n📸 Analyzing Share photos visually (sampling {max_photos})...")

    if not GEMINI_API_KEY:
        print("   ⚠️  No Gemini API key, skipping visual analysis")
        return None

    # Sample photos that exist
    available_photos = []
    for path in list(share_examples.keys())[:max_photos * 2]:  # Try 2x to account for missing
        p = Path(path)
        if p.exists():
            available_photos.append(p)
            if len(available_photos) >= max_photos:
                break

    if not available_photos:
        print("   ⚠️  No Share photos found")
        return None

    print(f"   Loading {len(available_photos)} Share photos...")

    # Load images
    images = []
    for path in available_photos:
        try:
            img = fix_image_orientation(path, max_size=800)
            if img:
                images.append(img)
        except Exception as e:
            print(f"   ⚠️  Could not load {path.name}: {e}")

    if not images:
        print("   ⚠️  No images loaded successfully")
        return None

    prompt = f"""Analyze these {len(images)} photos that the user CHOSE TO SHARE.

What visual patterns do you see? What makes these photos share-worthy?

Consider:
- Subject matter and composition
- Expressions and emotions
- Interactions between people
- Technical quality
- Framing and perspective
- Common themes or patterns

Respond with JSON:
{{
    "visual_patterns": [
        "list of common visual characteristics",
        "e.g., baby and parent both visible and engaged"
    ],
    "compositional_patterns": [
        "framing, perspective, distance patterns",
        "e.g., medium-close shots, subjects well-framed"
    ],
    "expression_patterns": [
        "facial expressions and emotional qualities",
        "e.g., smiling, engaged, making eye contact"
    ],
    "interaction_patterns": [
        "patterns in how subjects interact",
        "e.g., parent-child eye contact, shared activity"
    ],
    "quality_baseline": "description of minimum technical quality",
    "summary": "2-3 sentence summary of what makes these photos share-worthy"
}}

Be specific and concrete. Look for patterns across ALL photos."""

    try:
        model = genai.GenerativeModel('gemini-3.8-flash')
        content = [prompt] + images

        import time
        time.sleep(0.5)

        response = model.generate_content(content)
        text = response.text.strip()

        # Parse JSON
        import re
        json_match = re.search(r'\{.*\}', text, re.DOTALL)
        if json_match:
            result = json.loads(json_match.group(0))
            print(f"   ✅ Extracted visual patterns from Share photos")
            return result
        else:
            print(f"   ⚠️  Could not parse visual analysis")
            print(f"   Response: {text[:300]}")
            return None

    except Exception as e:
        print(f"   ⚠️  Error analyzing Share photos: {e}")
        import traceback
        print(f"   {traceback.format_exc()[:300]}")
        return None

def analyze_storage_photos_visual(storage_examples, max_photos=15):
    """Use Gemini to analyze visual patterns in Storage (rejected) photos."""
    print(f"\n🗑️  Analyzing Storage photos visually (sampling {max_photos})...")

    if not GEMINI_API_KEY:
        print("   ⚠️  No Gemini API key, skipping visual analysis")
        return None

    # Sample photos that exist
    available_photos = []
    for path in list(storage_examples.keys())[:max_photos * 2]:
        p = Path(path)
        if p.exists():
            available_photos.append(p)
            if len(available_photos) >= max_photos:
                break

    if not available_photos:
        print("   ⚠️  No Storage photos found")
        return None

    print(f"   Loading {len(available_photos)} Storage photos...")

    # Load images
    images = []
    for path in available_photos:
        try:
            img = fix_image_orientation(path, max_size=800)
            if img:
                images.append(img)
        except Exception as e:
            print(f"   ⚠️  Could not load {path.name}: {e}")

    if not images:
        print("   ⚠️  No images loaded successfully")
        return None

    prompt = f"""Analyze these {len(images)} photos that the user REJECTED (Storage).

What makes these photos NOT share-worthy? What patterns do you see?

Consider:
- Technical issues (blur, lighting, framing)
- Expression problems
- Composition issues
- Subject visibility
- Emotional quality

Respond with JSON:
{{
    "rejection_patterns": [
        "list of common reasons for rejection",
        "e.g., can't see baby's face clearly"
    ],
    "technical_issues": [
        "technical problems that lead to rejection",
        "e.g., blurry, poor lighting, bad framing"
    ],
    "expression_issues": [
        "facial expression problems",
        "e.g., eyes closed, vacant expressions, looking away"
    ],
    "composition_issues": [
        "framing and composition problems",
        "e.g., too zoomed out, unclear subject, awkward angle"
    ],
    "summary": "2-3 sentence summary of what makes these photos rejectable"
}}

Be specific. Look for patterns across ALL photos."""

    try:
        model = genai.GenerativeModel('gemini-3.8-flash')
        content = [prompt] + images

        import time
        time.sleep(0.5)

        response = model.generate_content(content)
        text = response.text.strip()

        # Parse JSON
        import re
        json_match = re.search(r'\{.*\}', text, re.DOTALL)
        if json_match:
            result = json.loads(json_match.group(0))
            print(f"   ✅ Extracted patterns from Storage photos")
            return result
        else:
            print(f"   ⚠️  Could not parse visual analysis")
            print(f"   Response: {text[:300]}")
            return None

    except Exception as e:
        print(f"   ⚠️  Error analyzing Storage photos: {e}")
        import traceback
        print(f"   {traceback.format_exc()[:300]}")
        return None

def generate_comprehensive_profile(reasoning_analysis, share_visual, storage_visual):
    """Generate comprehensive taste profile from all analyses."""
    print("\n📝 Generating comprehensive taste profile...")

    if not GEMINI_API_KEY:
        print("   ⚠️  No Gemini API key, generating basic profile")
        return generate_basic_profile(reasoning_analysis, share_visual, storage_visual)

    # Combine all insights
    prompt = f"""Synthesize these analyses into a comprehensive photo sorting taste profile.

**REASONING TEXT ANALYSIS:**
{json.dumps(reasoning_analysis, indent=2) if reasoning_analysis else "Not available"}

**SHARE PHOTOS VISUAL ANALYSIS:**
{json.dumps(share_visual, indent=2) if share_visual else "Not available"}

**STORAGE PHOTOS VISUAL ANALYSIS:**
{json.dumps(storage_visual, indent=2) if storage_visual else "Not available"}

Create a comprehensive taste profile as JSON:
{{
    "share_philosophy": "2-3 sentence philosophy of what makes photos share-worthy for this user",
    "top_priorities": [
        "5-7 most important factors in order of priority",
        "Be specific and actionable"
    ],
    "share_criteria": {{
        "must_have": ["non-negotiable requirements"],
        "highly_valued": ["strongly preferred attributes"],
        "bonus_points": ["nice-to-have qualities"]
    }},
    "reject_criteria": {{
        "deal_breakers": ["automatic rejection reasons"],
        "negative_factors": ["qualities that count against a photo"]
    }},
    "contextual_preferences": {{
        "parent_expressions": "how parent/adult expressions factor in",
        "interaction_quality": "preferences for interaction vs solo shots",
        "technical_threshold": "minimum technical quality expectations",
        "emotional_content": "emotional qualities valued",
        "burst_philosophy": "approach to selecting from burst photos"
    }},
    "specific_guidance": [
        "specific, actionable guidance for photo evaluation",
        "e.g., 'Both parent and child should have engaged expressions'",
        "e.g., 'Reject if can't see baby's face clearly, even if parent is well-framed'"
    ]
}}

Synthesize insights across all three analyses. Be specific, concrete, and actionable."""

    try:
        model = genai.GenerativeModel('gemini-3.8-flash')
        response = model.generate_content(prompt)
        text = response.text.strip()

        # Parse JSON
        import re
        # Handle markdown code fences
        text_cleaned = re.sub(r'^```json\s*\n', '', text, flags=re.MULTILINE)
        text_cleaned = re.sub(r'^```\s*\n', '', text_cleaned, flags=re.MULTILINE)
        text_cleaned = re.sub(r'\n```\s*$', '', text_cleaned, flags=re.MULTILINE)

        json_match = re.search(r'\{.*\}', text_cleaned, re.DOTALL)
        if json_match:
            result = json.loads(json_match.group(0))
            print(f"   ✅ Generated comprehensive profile")
            return result
        else:
            print(f"   ⚠️  Could not parse synthesis")
            print(f"   Response: {text[:500]}")
            return None

    except Exception as e:
        print(f"   ⚠️  Error generating profile: {e}")
        return None

def generate_basic_profile(reasoning_analysis, share_visual, storage_visual):
    """Generate basic profile without Gemini synthesis."""
    print("   Generating basic profile from available data...")

    profile = {
        "share_philosophy": "Photos that capture genuine moments of connection and emotion, especially parent-child interaction, with good technical quality.",
        "top_priorities": [],
        "share_criteria": {
            "must_have": ["Baby's face clearly visible", "Good technical quality"],
            "highly_valued": [],
            "bonus_points": []
        },
        "reject_criteria": {
            "deal_breakers": [],
            "negative_factors": []
        },
        "contextual_preferences": {
            "parent_expressions": "Parent expressions matter as much as baby's",
            "interaction_quality": "Prefer interaction shots over solo baby photos",
            "technical_threshold": "Photos must be sharp and well-lit",
            "emotional_content": "Value emotional moments and expressions",
            "burst_philosophy": "Only keep 1-2 best from each burst"
        },
        "specific_guidance": []
    }

    # Extract from reasoning analysis
    if reasoning_analysis:
        profile["top_priorities"].extend(reasoning_analysis.get("valued_qualities", []))
        profile["share_criteria"]["highly_valued"].extend(reasoning_analysis.get("specific_preferences", []))
        profile["reject_criteria"]["deal_breakers"].extend(reasoning_analysis.get("reject_criteria", []))

    # Extract from visual analyses
    if share_visual:
        profile["share_criteria"]["highly_valued"].extend(share_visual.get("visual_patterns", []))
        profile["share_criteria"]["bonus_points"].extend(share_visual.get("interaction_patterns", []))

    if storage_visual:
        profile["reject_criteria"]["deal_breakers"].extend(storage_visual.get("rejection_patterns", []))
        profile["reject_criteria"]["negative_factors"].extend(storage_visual.get("technical_issues", []))

    return profile

def save_profile(profile):
    """Save generated profile to file."""
    if not profile:
        print("\n❌ No profile generated")
        return False

    print(f"\n💾 Saving profile to {OUTPUT_FILE}...")

    with open(OUTPUT_FILE, 'w') as f:
        json.dump(profile, f, indent=2)

    print(f"   ✅ Saved to {OUTPUT_FILE}")

    # Print preview
    print("\n" + "="*70)
    print("GENERATED TASTE PROFILE PREVIEW")
    print("="*70)

    if "share_philosophy" in profile:
        print(f"\n📜 Philosophy:\n   {profile['share_philosophy']}")

    if "top_priorities" in profile and profile["top_priorities"]:
        print(f"\n🎯 Top Priorities:")
        for i, priority in enumerate(profile["top_priorities"][:5], 1):
            print(f"   {i}. {priority}")

    if "share_criteria" in profile:
        must_have = profile["share_criteria"].get("must_have", [])
        if must_have:
            print(f"\n✅ Must Have:")
            for item in must_have[:3]:
                print(f"   • {item}")

    if "reject_criteria" in profile:
        deal_breakers = profile["reject_criteria"].get("deal_breakers", [])
        if deal_breakers:
            print(f"\n❌ Deal Breakers:")
            for item in deal_breakers[:3]:
                print(f"   • {item}")

    print("\n" + "="*70)

    return True

def main():
    print("="*70)
    print("TASTE PROFILE GENERATOR")
    print("="*70)

    if not GEMINI_API_KEY:
        print("\n⚠️  WARNING: No GEMINI_API_KEY found!")
        print("   Will generate basic profile without visual analysis")
        input("\nPress Enter to continue or Ctrl+C to cancel...")

    # Load training data
    result = load_training_data()
    if not result:
        return

    share_examples, storage_examples = result

    # Analyze reasoning text
    reasoning_analysis = analyze_reasoning_text(share_examples, storage_examples)

    # Analyze Share photos visually
    share_visual = analyze_share_photos_visual(share_examples, max_photos=15)

    # Analyze Storage photos visually
    storage_visual = analyze_storage_photos_visual(storage_examples, max_photos=15)

    # Generate comprehensive profile
    profile = generate_comprehensive_profile(reasoning_analysis, share_visual, storage_visual)

    # Save profile
    if save_profile(profile):
        print(f"\n✅ Complete! Edit {OUTPUT_FILE} to refine your preferences.")
        print(f"   Then update taste_classify_gemini_v4.py to load this file.")
    else:
        print("\n❌ Failed to generate profile")

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\n⚠️  Interrupted by user")
    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()
