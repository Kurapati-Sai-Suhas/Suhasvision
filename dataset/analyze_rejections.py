import os
from collections import Counter

LOG_FILE = "pipeline_rejections.log"

def analyze_rejections():
    if not os.path.exists(LOG_FILE):
        print("No rejections log found.")
        return

    total_entries = 0
    categories = Counter()
    
    with open(LOG_FILE, "r", encoding="utf-8") as f:
        for line in f:
            parts = line.strip().split(" | ")
            if len(parts) >= 3:
                category = parts[2].strip()
                categories[category] += 1
                total_entries += 1

    print("=" * 40)
    print("PIPELINE REJECTIONS ANALYSIS")
    print("=" * 40)
    print(f"Total Logged Events: {total_entries}\n")
    
    for cat, count in categories.most_common():
        pct = (count / total_entries) * 100
        print(f"- {cat}: {count} ({pct:.1f}%)")
        
    print("\nNote: 'LOW_VISIBILITY' and 'TOPOLOGY_INVERTED' denote frame-level failures that may have been salvaged via interpolation if they did not hit the hard rejection thresholds.")
    
if __name__ == "__main__":
    analyze_rejections()
