// Rich mock dataset for SuhasVision UI. Fully static, no backend.

export const CURRENT_COACH = {
  id: "coach-suhas",
  name: "Coach Suhas Menon",
  title: "Head Batting Coach · Deccan Cricket Academy",
  email: "suhas@deccan.cricket",
  avatar:
    "https://images.unsplash.com/photo-1544005313-94ddf0286df2?crop=faces&fit=crop&w=256&h=256",
  academy: "Deccan Cricket Academy",
  activePlayers: 47,
  videosThisWeek: 32,
  overrideRate: 0.18,
};

export const CURRENT_LEARNER = {
  id: "learner-arya",
  name: "Arya Patel",
  age: 17,
  role: "Top-order batter · Right-hand",
  academy: "Deccan Cricket Academy · U19",
  email: "arya.patel@deccan.cricket",
  avatar:
    "https://images.unsplash.com/photo-1607746882042-944635dfe10e?crop=faces&fit=crop&w=256&h=256",
  streak: 24,
  totalPoints: 4820,
  currentTier: "Gold IV",
  nextTier: "Platinum I",
  tierProgress: 68,
  rank: 3,
  academyRank: 3,
  weeklyGoal: 5,
  weeklyProgress: 4,
};

export const BADGES = [
  {
    id: "b1",
    name: "First Fifty",
    description: "Logged 50 practice sessions",
    icon: "trophy",
    tone: "gold",
    unlocked: true,
    unlockedAt: "Jan 12, 2026",
  },
  {
    id: "b2",
    name: "Ironwrist",
    description: "20 sessions with balance score > 85",
    icon: "shield",
    tone: "emerald",
    unlocked: true,
    unlockedAt: "Jan 24, 2026",
  },
  {
    id: "b3",
    name: "Cover Drive Artist",
    description: "Technique score above 90 on cover drives",
    icon: "sparkles",
    tone: "emerald",
    unlocked: true,
    unlockedAt: "Feb 02, 2026",
  },
  {
    id: "b4",
    name: "Streak Keeper",
    description: "20-day practice streak",
    icon: "flame",
    tone: "gold",
    unlocked: true,
    unlockedAt: "Feb 08, 2026",
  },
  {
    id: "b5",
    name: "Power Hitter",
    description: "Reach a power score of 95",
    icon: "zap",
    tone: "locked",
    unlocked: false,
    progress: 78,
  },
  {
    id: "b6",
    name: "Century Mindset",
    description: "Complete 100 uploads",
    icon: "award",
    tone: "locked",
    unlocked: false,
    progress: 62,
  },
];

export const PROGRESS_TIMELINE = [
  { week: "W1", balance: 62, power: 58, technique: 60, overall: 60 },
  { week: "W2", balance: 65, power: 61, technique: 62, overall: 63 },
  { week: "W3", balance: 68, power: 63, technique: 66, overall: 66 },
  { week: "W4", balance: 71, power: 66, technique: 69, overall: 69 },
  { week: "W5", balance: 74, power: 70, technique: 72, overall: 72 },
  { week: "W6", balance: 76, power: 73, technique: 74, overall: 74 },
  { week: "W7", balance: 79, power: 76, technique: 78, overall: 78 },
  { week: "W8", balance: 82, power: 78, technique: 80, overall: 80 },
  { week: "W9", balance: 84, power: 82, technique: 83, overall: 83 },
  { week: "W10", balance: 86, power: 84, technique: 85, overall: 85 },
  { week: "W11", balance: 88, power: 86, technique: 87, overall: 87 },
  { week: "W12", balance: 90, power: 88, technique: 89, overall: 89 },
];

export const LEARNER_UPLOADS = [
  {
    id: "u1",
    title: "Cover drive — front-foot session",
    date: "Feb 09, 2026",
    duration: "0:42",
    thumbnail:
      "https://images.pexels.com/photos/3628912/pexels-photo-3628912.jpeg?auto=compress&cs=tinysrgb&w=800",
    scores: { balance: 88, power: 84, technique: 91 },
    overall: 88,
    status: "reviewed",
    feedbackId: "f1",
  },
  {
    id: "u2",
    title: "Pull shot — bouncer drill",
    date: "Feb 07, 2026",
    duration: "0:38",
    thumbnail:
      "https://images.pexels.com/photos/163452/basketball-dunk-blue-game-163452.jpeg?auto=compress&cs=tinysrgb&w=800",
    scores: { balance: 82, power: 90, technique: 79 },
    overall: 84,
    status: "reviewed",
    feedbackId: "f2",
  },
  {
    id: "u3",
    title: "Straight drive — throwdowns",
    date: "Feb 05, 2026",
    duration: "0:51",
    thumbnail:
      "https://images.pexels.com/photos/36230651/pexels-photo-36230651.jpeg?auto=compress&cs=tinysrgb&w=800",
    scores: { balance: 85, power: 81, technique: 87 },
    overall: 84,
    status: "pending",
  },
  {
    id: "u4",
    title: "Late cut — spin machine",
    date: "Feb 02, 2026",
    duration: "0:33",
    thumbnail:
      "https://images.unsplash.com/photo-1531415074968-036ba1b575da?w=800",
    scores: { balance: 79, power: 71, technique: 83 },
    overall: 78,
    status: "reviewed",
    feedbackId: "f3",
  },
];

export const FEEDBACK_INBOX = [
  {
    id: "f1",
    videoId: "u1",
    videoTitle: "Cover drive — front-foot session",
    coachName: "Coach Suhas Menon",
    coachAvatar: CURRENT_COACH.avatar,
    date: "Feb 10, 2026",
    aiSummary:
      "AI detected slightly delayed front-foot commit. Bat swing is on line but head is falling toward off-stump on contact.",
    coachOverride: {
      applied: true,
      previousScore: 84,
      newScore: 88,
      note: "AI under-scored technique — front elbow position looked textbook against 130+ kph feeds.",
    },
    recommendations: [
      "Continue front-foot commit drills — 3 sets of 20 with the sponge ball.",
      "Focus on keeping head steady until the ball is under the eye.",
      "Try the 'stand tall' cue on the 2nd bounce delivery.",
    ],
    unread: true,
  },
  {
    id: "f2",
    videoId: "u2",
    videoTitle: "Pull shot — bouncer drill",
    coachName: "Coach Suhas Menon",
    coachAvatar: CURRENT_COACH.avatar,
    date: "Feb 08, 2026",
    aiSummary:
      "Excellent power generation via rear hip rotation. Bat face is slightly closed on contact leading to lower trajectory.",
    coachOverride: null,
    recommendations: [
      "Practice pulling with the top-hand slightly softer to open the bat face.",
      "Work on picking up length in the first 0.2 seconds — reaction drill 2B.",
    ],
    unread: false,
  },
  {
    id: "f3",
    videoId: "u4",
    videoTitle: "Late cut — spin machine",
    coachName: "Coach Suhas Menon",
    coachAvatar: CURRENT_COACH.avatar,
    date: "Feb 03, 2026",
    aiSummary:
      "Late cut selection is good, but the balance score suggests the weight is moving forward too early.",
    coachOverride: {
      applied: true,
      previousScore: 82,
      newScore: 78,
      note: "AI over-scored — footwork is committing before ball release. Revised.",
    },
    recommendations: [
      "Hold the trigger movement until pitch — shadow drill for 10 minutes daily.",
      "Split-vision drill against left-arm spin.",
    ],
    unread: false,
  },
];

// COACH view — review queue (submissions from various learners)
export const REVIEW_QUEUE = [
  {
    id: "rq1",
    learner: "Arya Patel",
    avatar: CURRENT_LEARNER.avatar,
    tier: "Gold IV",
    videoTitle: "Straight drive — throwdowns",
    submittedAt: "23 minutes ago",
    duration: "0:51",
    thumbnail:
      "https://images.pexels.com/photos/36230651/pexels-photo-36230651.jpeg?auto=compress&cs=tinysrgb&w=800",
    aiScore: 84,
    urgency: "high",
    scores: { balance: 85, power: 81, technique: 87 },
  },
  {
    id: "rq2",
    learner: "Kabir Ranjan",
    avatar:
      "https://images.unsplash.com/photo-1603415526960-f7e0328c63b1?crop=faces&fit=crop&w=256&h=256",
    tier: "Silver II",
    videoTitle: "Sweep shot — left-arm spin",
    submittedAt: "1 hour ago",
    duration: "0:47",
    thumbnail:
      "https://images.unsplash.com/photo-1531415074968-036ba1b575da?w=800",
    aiScore: 76,
    urgency: "medium",
    scores: { balance: 78, power: 72, technique: 78 },
  },
  {
    id: "rq3",
    learner: "Isha Deshmukh",
    avatar:
      "https://images.unsplash.com/photo-1544005313-94ddf0286df2?crop=faces&fit=crop&w=256&h=256",
    tier: "Platinum I",
    videoTitle: "Cover drive — new-ball session",
    submittedAt: "3 hours ago",
    duration: "1:04",
    thumbnail:
      "https://images.pexels.com/photos/3628912/pexels-photo-3628912.jpeg?auto=compress&cs=tinysrgb&w=800",
    aiScore: 91,
    urgency: "low",
    scores: { balance: 92, power: 88, technique: 93 },
  },
  {
    id: "rq4",
    learner: "Rohit Bhaskar",
    avatar:
      "https://images.unsplash.com/photo-1607746882042-944635dfe10e?crop=faces&fit=crop&w=256&h=256",
    tier: "Gold II",
    videoTitle: "On-drive — pace machine 140kph",
    submittedAt: "5 hours ago",
    duration: "0:39",
    thumbnail:
      "https://images.pexels.com/photos/163452/basketball-dunk-blue-game-163452.jpeg?auto=compress&cs=tinysrgb&w=800",
    aiScore: 82,
    urgency: "medium",
    scores: { balance: 84, power: 82, technique: 80 },
  },
  {
    id: "rq5",
    learner: "Meera Iyer",
    avatar:
      "https://images.unsplash.com/photo-1517841905240-472988babdf9?crop=faces&fit=crop&w=256&h=256",
    tier: "Gold III",
    videoTitle: "Backfoot punch — bounce drill",
    submittedAt: "Yesterday",
    duration: "0:28",
    thumbnail:
      "https://images.pexels.com/photos/3628912/pexels-photo-3628912.jpeg?auto=compress&cs=tinysrgb&w=800",
    aiScore: 79,
    urgency: "low",
    scores: { balance: 80, power: 77, technique: 79 },
  },
];

// COACH view — leaderboard of players
export const PLAYER_RANKINGS = [
  {
    rank: 1,
    id: "p-isha",
    name: "Isha Deshmukh",
    tier: "Platinum I",
    avatar:
      "https://images.unsplash.com/photo-1544005313-94ddf0286df2?crop=faces&fit=crop&w=256&h=256",
    aiScore: 93,
    streak: 41,
    sessions: 128,
    trend: "up",
    trendValue: 4.2,
    specialty: "Cover drive",
    highlight: "Selected for state camp",
  },
  {
    rank: 2,
    id: "p-vihaan",
    name: "Vihaan Shetty",
    tier: "Platinum I",
    avatar:
      "https://images.unsplash.com/photo-1603415526960-f7e0328c63b1?crop=faces&fit=crop&w=256&h=256",
    aiScore: 91,
    streak: 33,
    sessions: 112,
    trend: "up",
    trendValue: 2.1,
    specialty: "Pull shot",
    highlight: "Unbeaten 132 in U19 trial",
  },
  {
    rank: 3,
    id: "p-arya",
    name: "Arya Patel",
    tier: "Gold IV",
    avatar: CURRENT_LEARNER.avatar,
    aiScore: 89,
    streak: 24,
    sessions: 96,
    trend: "up",
    trendValue: 3.4,
    specialty: "Straight drive",
    highlight: "Personal-best 12-week arc",
  },
  {
    rank: 4,
    id: "p-rohit",
    name: "Rohit Bhaskar",
    tier: "Gold II",
    avatar:
      "https://images.unsplash.com/photo-1607746882042-944635dfe10e?crop=faces&fit=crop&w=256&h=256",
    aiScore: 84,
    streak: 18,
    sessions: 71,
    trend: "up",
    trendValue: 1.6,
    specialty: "On-drive",
    highlight: "Improved power +7 this month",
  },
  {
    rank: 5,
    id: "p-meera",
    name: "Meera Iyer",
    tier: "Gold III",
    avatar:
      "https://images.unsplash.com/photo-1517841905240-472988babdf9?crop=faces&fit=crop&w=256&h=256",
    aiScore: 82,
    streak: 27,
    sessions: 84,
    trend: "flat",
    trendValue: 0.2,
    specialty: "Backfoot punch",
    highlight: "Most consistent balance score",
  },
  {
    rank: 6,
    id: "p-kabir",
    name: "Kabir Ranjan",
    tier: "Silver II",
    avatar:
      "https://images.unsplash.com/photo-1603415526960-f7e0328c63b1?crop=faces&fit=crop&w=256&h=256",
    aiScore: 78,
    streak: 12,
    sessions: 52,
    trend: "up",
    trendValue: 2.9,
    specialty: "Sweep shot",
    highlight: "Fastest improvement rate",
  },
  {
    rank: 7,
    id: "p-tara",
    name: "Tara Nair",
    tier: "Silver III",
    avatar:
      "https://images.unsplash.com/photo-1531123897727-8f129e1688ce?crop=faces&fit=crop&w=256&h=256",
    aiScore: 74,
    streak: 9,
    sessions: 44,
    trend: "down",
    trendValue: -0.8,
    specialty: "Late cut",
    highlight: "Working on trigger movement",
  },
];

// COACH view — talent scouting (top prospects with more context)
export const TALENT_PROSPECTS = [
  {
    id: "p-isha",
    name: "Isha Deshmukh",
    tier: "Platinum I",
    age: 18,
    hometown: "Pune, MH",
    avatar:
      "https://images.unsplash.com/photo-1544005313-94ddf0286df2?crop=faces&fit=crop&w=400&h=400",
    aiScore: 93,
    peakScore: 95,
    style: "Right-hand, top-order",
    strengths: ["Cover drive", "Backfoot defence", "Situational awareness"],
    weaknesses: ["Sweep vs left-arm spin"],
    signature: "Textbook front-foot technique with elite balance metrics",
    nextLevelReady: true,
    contactedAt: null,
  },
  {
    id: "p-vihaan",
    name: "Vihaan Shetty",
    tier: "Platinum I",
    age: 19,
    hometown: "Mangalore, KA",
    avatar:
      "https://images.unsplash.com/photo-1603415526960-f7e0328c63b1?crop=faces&fit=crop&w=400&h=400",
    aiScore: 91,
    peakScore: 93,
    style: "Right-hand, aggressive middle-order",
    strengths: ["Pull shot", "Power hitting", "Bounce handling"],
    weaknesses: ["Consistency vs full-length"],
    signature: "Explosive bat-speed with elite rear-hip rotation",
    nextLevelReady: true,
    contactedAt: "Feb 05, 2026",
  },
  {
    id: "p-arya",
    name: "Arya Patel",
    tier: "Gold IV",
    age: 17,
    hometown: "Ahmedabad, GJ",
    avatar: CURRENT_LEARNER.avatar,
    aiScore: 89,
    peakScore: 91,
    style: "Right-hand, top-order",
    strengths: ["Straight drive", "Composure", "Front-foot commit"],
    weaknesses: ["Sweep on turning tracks"],
    signature: "Best 12-week improvement arc in the academy",
    nextLevelReady: true,
    contactedAt: null,
  },
];

export const COACH_METRICS = [
  {
    label: "Videos in queue",
    value: 12,
    delta: "+4 today",
    tone: "emerald",
    testId: "metric-queue",
  },
  {
    label: "Active players",
    value: 47,
    delta: "3 new this week",
    tone: "gold",
    testId: "metric-active",
  },
  {
    label: "Avg academy score",
    value: 82.4,
    delta: "+1.6 vs last week",
    tone: "emerald",
    testId: "metric-avg",
  },
  {
    label: "AI overrides",
    value: "18%",
    delta: "Healthy calibration",
    tone: "muted",
    testId: "metric-override",
  },
];

export const COACH_SESSIONS = [
  {
    time: "07:00",
    title: "Morning nets · U19 squad",
    location: "Ground A",
    attendees: 12,
  },
  {
    time: "10:30",
    title: "1-on-1 · Isha Deshmukh",
    location: "Indoor Bay 2",
    attendees: 1,
  },
  {
    time: "14:00",
    title: "Video Room · Weekly review",
    location: "Room 204",
    attendees: 8,
  },
  {
    time: "17:00",
    title: "Power-hitting clinic",
    location: "Ground B",
    attendees: 6,
  },
];

export const BIOMECHANICS_SNAPSHOT = {
  balance: {
    score: 85,
    trend: "+3",
    breakdown: [
      { label: "Head position", value: 91 },
      { label: "Base stability", value: 82 },
      { label: "Weight transfer", value: 84 },
    ],
  },
  power: {
    score: 81,
    trend: "+1",
    breakdown: [
      { label: "Hip rotation", value: 86 },
      { label: "Bat swing arc", value: 79 },
      { label: "Follow-through", value: 78 },
    ],
  },
  technique: {
    score: 87,
    trend: "+4",
    breakdown: [
      { label: "Bat-path angle", value: 89 },
      { label: "Front-elbow position", value: 88 },
      { label: "Contact timing", value: 85 },
    ],
  },
};

export const RECENT_ACTIVITY = [
  {
    id: "a1",
    icon: "video",
    text: "Isha Deshmukh submitted 'Cover drive — new-ball session'",
    time: "3h ago",
  },
  {
    id: "a2",
    icon: "star",
    text: "Vihaan Shetty crossed a peak score of 93 for the first time",
    time: "6h ago",
  },
  {
    id: "a3",
    icon: "trend",
    text: "Academy avg balance score improved to 84.1 (+1.3)",
    time: "Yesterday",
  },
  {
    id: "a4",
    icon: "mail",
    text: "Invite sent to Vihaan Shetty for state-level trial",
    time: "2 days ago",
  },
];
