"""Synthetic client-advisor conversations with exact gold answers.

Each client has a fact timeline (risk tolerance, income, employer, city, savings plan, contributions, goals,
family) that changes over about 25 dated sessions. Sessions are rendered from templates, padded with
generic advice and small talk, and every fact-bearing turn is recorded, so each question knows its gold
answer, the turns that support it, and the exact wrong answers its traps produce.

All people, firms and employers are fictional. No question depends on real account rules or limits: every
gold value is something a client or advisor said in the conversation.
"""

from __future__ import annotations

import datetime as dt
import hashlib
import json
import random
from dataclasses import asdict, dataclass, field
from pathlib import Path

FIRM = "Tidewell Wealth"
NOTICE = "Synthetic conversation for research. All people, firms and employers are fictional."
RISK = ["conservative", "moderately conservative", "balanced", "growth", "aggressive"]
CONTACT = ["email", "phone call", "text message", "video call"]
CITIES = ["Vancouver", "Burnaby", "Richmond", "Surrey", "North Vancouver", "Coquitlam", "Victoria", "Kelowna",
          "Langley", "New Westminster", "Nanaimo", "Squamish"]
SECTORS = ["tobacco", "weapons manufacturing", "oil sands", "gambling", "coal mining", "fast fashion"]
EMPLOYERS = ["Harbourline Logistics", "Cedarpoint Software", "Northshore Dental Group", "Fraser Valley Health",
             "Bluewater Engineering", "Granite Peak Surveying", "Seawall Media", "Larchwood Construction",
             "Coastline Analytics", "Evergreen Bakeries", "Saltspring Design Studio", "Aurora Cell Labs",
             "Quayside Freight", "Ironbark Robotics", "Moss Lake Veterinary", "Brightwater Schools"]
ROLES = ["project manager", "software developer", "dental hygienist", "nurse", "civil engineer", "surveyor",
         "video editor", "site supervisor", "data analyst", "operations lead", "graphic designer", "lab scientist"]
CLIENTS = [("Maya", "Chen", "f"), ("Arjun", "Patel", "m"), ("Sofia", "Rossi", "f"), ("Liam", "O'Connor", "m"),
           ("Hana", "Kim", "f"), ("Daniel", "Okafor", "m"), ("Leila", "Haddad", "f"), ("Ethan", "Tremblay", "m"),
           ("Mei", "Wong", "f"), ("Gabriel", "Silva", "m"), ("Aisha", "Rahman", "f"), ("Noah", "Fraser", "m")]
SPOUSES = {"f": ["Ryan", "Kevin", "Omar", "Mateo", "Jun", "Samuel"], "m": ["Priya", "Grace", "Elena", "Nadia", "Chloe", "Yuna"]}
KIDS = {"f": ["Ava", "Isla", "Nora", "Zara"], "m": ["Leo", "Kai", "Felix", "Omar"]}
ADVISORS = ["Dana Whitfield", "Marcus Lee", "Elena Petrova"]
DEV_CLIENTS = 3  # the first three clients form the dev split
N_SESSIONS = 25
WEEKDAYS = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
NUM_WORDS = {"zero": 0, "no": 0, "one": 1, "two": 2, "three": 3, "four": 4}


@dataclass
class Turn:
    tid: str
    session: int
    date: str
    speaker: str      # "client" | "advisor"
    name: str
    text: str

    def line(self) -> str:
        return f"[{self.date}] {self.name}: {self.text}"


@dataclass
class Client:
    cid: str
    name: str
    advisor: str
    split: str
    today: str
    sessions: list = field(default_factory=list)   # list of lists of Turn
    questions: list = field(default_factory=list)

    @property
    def turns(self) -> list[Turn]:
        return [t for s in self.sessions for t in s]

    def to_json(self) -> dict:
        d = asdict(self)
        return d

    @classmethod
    def from_json(cls, d: dict) -> "Client":
        c = cls(d["cid"], d["name"], d["advisor"], d["split"], d["today"])
        c.sessions = [[Turn(**t) for t in s] for s in d["sessions"]]
        c.questions = d["questions"]
        return c


def money(v: int) -> str:
    return f"${v:,}"


def _iso(d: dt.date) -> str:
    return d.isoformat()


def _last_weekday(d: dt.date, wd: int) -> dt.date:
    """The most recent given weekday strictly before d."""
    delta = (d.weekday() - wd) % 7 or 7
    return d - dt.timedelta(days=delta)


# ---------------------------------------------------------------- filler

ADVICE = [
    "Quick update on the portfolio: since our last meeting it is up about {p}%, mostly from the equity side. "
    "Bonds were roughly flat, which is what we expect them to do when rates are not moving much.",
    "One thing I always remind clients about is rebalancing. When one part of the portfolio runs ahead, we trim "
    "it back to the target mix so the overall risk stays where you chose it. It is a boring habit, but it works.",
    "I also want to flag fees. Your blended fund costs come to about {f}% a year. That is reasonable, but I check "
    "it every year to make sure we are not paying for something we could get more cheaply.",
    "Markets were a bit choppy this month. Your accounts dipped about {d}% at one point and then recovered most of "
    "it. Nothing to act on, but I would rather you hear it from me than from the headlines.",
    "Something to keep in mind is the emergency fund. We usually like to see a few months of expenses in cash or a "
    "high-interest savings account before money goes into longer-term investments.",
    "I reviewed the beneficiary designations on file. It is worth checking them whenever there is a big life "
    "change, so let me know if anything in the family situation shifts.",
    "On taxes, the usual reminder: keep your slips together when they arrive, and send me copies if you want me to "
    "look at how the registered and non-registered accounts fit together.",
    "We also looked at cash drag. There was about {c}% of the portfolio sitting in cash after the last dividend "
    "payments, so I swept it back into the core funds.",
    "I ran the long-term projection again with the current numbers. It still looks on track, although these "
    "projections move around a lot, so I treat them as a rough guide rather than a promise.",
    "A note on currency: some of the global funds hold US assets, so the Canadian dollar moving by {x}% shows up in "
    "the returns even when the underlying companies did nothing special.",
    "On insurance, it is worth reviewing coverage every couple of years. Group plans through work can change "
    "quietly, and the amount that made sense a few years ago may not match the household today.",
    "I like to separate money by job: cash for the next year or two, a middle bucket for medium-term plans, and "
    "the long-term portfolio. It makes it easier to stay calm when markets move.",
    "We talked to the fund company about their new share class. The difference works out to roughly {f}% a year, "
    "so I will switch you over when the paperwork is ready. There is no tax cost to the switch.",
    "A general point on behaviour: the biggest risk for most investors is selling after a drop. Having a written "
    "plan helps, which is why I keep coming back to the one we agreed on.",
    "I also checked the asset location. Interest-paying holdings generally sit better in the registered accounts, "
    "while the equity funds can sit in the taxable account. We are mostly set up that way already.",
    "For estate planning, it may be worth meeting a lawyer at some point to update wills and powers of attorney. "
    "I can suggest a few names if that would help.",
]
SMALLTALK = [
    ("How have things been since we last spoke?", "Busy, honestly. Work has been a lot, but we are managing."),
    ("Did you get a break at all over the last few weeks?", "A little. We stayed close to home and mostly rested."),
    ("How is the family doing?", "Everyone is well. The usual chaos, but good chaos."),
    ("Have you been following the news on the economy?", "Only the headlines. It mostly makes me want to check my accounts less."),
    ("Anything new at work?", "Not much. A few new projects, nothing dramatic."),
    ("Did the weather cooperate this month?", "Not really, lots of rain. We spent more time indoors than we wanted."),
    ("Are you still doing the weekend hikes?", "When we can. It is hard to find the time lately."),
    ("How did the move of your office desk work out?", "Fine, I am closer to the window now, which helps."),
]
OPEN_A = ["Good to see you again, {n}. Let's go through what is new.", "Thanks for making the time, {n}. Shall we start with any updates?",
          "Hi {n}, hope the week has been kind to you. Let's catch up."]
OPEN_C = ["Thanks, good to talk.", "Sure, happy to.", "Hi, yes, let's do it."]
CLOSE_A = ["I'll send a short summary by your preferred channel. Talk soon.", "That covers it for today. I'll follow up with the paperwork.",
           "Great, I'll make the changes we discussed and check in next time."]
CLOSE_C = ["Thanks, that was helpful.", "Perfect, talk soon.", "Sounds good, thank you."]


class Builder:
    """Collects one client's sessions, the turns behind every fact, and the fact timeline."""

    def __init__(self, rng: random.Random, cid: str, name: str, advisor: str, dates: list[dt.date]) -> None:
        self.rng = rng
        self.cid = cid
        self.name = name
        self.first = name.split()[0]
        self.advisor = advisor
        self.dates = dates
        self.plan: dict[int, list] = {i: [] for i in range(len(dates))}
        self.sessions: list[list[Turn]] = []
        self.evidence: dict[str, list[str]] = {}   # fact key -> turn ids

    def at(self, s: int, fn, key: str | None = None) -> None:
        self.plan[s].append((fn, key))

    def render(self) -> None:
        for s, date in enumerate(self.dates):
            turns: list[tuple[str, str]] = []
            keys: list[tuple[int, int, str]] = []
            turns += [("advisor", self.rng.choice(OPEN_A).format(n=self.first)), ("client", self.rng.choice(OPEN_C))]
            blocks = [("fact", fn, key) for fn, key in self.plan[s]]
            blocks += [("advice", None, None)] * self.rng.randint(3, 4) + [("talk", None, None)] * self.rng.randint(1, 2)
            self.rng.shuffle(blocks)
            for kind, fn, key in blocks:
                if kind == "fact":
                    start = len(turns)
                    turns += fn(date)
                    if key:
                        keys.append((start, len(turns), key))
                elif kind == "advice":
                    t = " ".join(self.rng.sample(ADVICE, 2)).format(p=round(self.rng.uniform(0.4, 3.9), 1), f=round(self.rng.uniform(0.3, 0.9), 2),
                                                       d=round(self.rng.uniform(1.0, 4.5), 1), c=round(self.rng.uniform(1, 6), 1),
                                                       x=round(self.rng.uniform(1, 4), 1))
                    turns += [("advisor", t), ("client", self.rng.choice(["Makes sense.", "Okay, good to know.", "Thanks for explaining that.",
                                                                          "Got it.", "That's reassuring."]))]
                else:
                    a, c = self.rng.choice(SMALLTALK)
                    turns += [("advisor", a), ("client", c)]
            turns += [("advisor", self.rng.choice(CLOSE_A)), ("client", self.rng.choice(CLOSE_C))]
            out = []
            for i, (sp, text) in enumerate(turns):
                out.append(Turn(f"{self.cid}-s{s + 1:02d}-t{i + 1:02d}", s + 1, _iso(date), sp,
                                self.name if sp == "client" else f"{self.advisor} (advisor)", text))
            for a, b, key in keys:
                self.evidence.setdefault(key, []).extend(t.tid for t in out[a:b])
            self.sessions.append(out)


# ---------------------------------------------------------------- one client

def _session_dates(rng: random.Random) -> list[dt.date]:
    d = dt.date(2025, 1, 6) + dt.timedelta(days=rng.randint(0, 45))
    out = []
    for _ in range(N_SESSIONS):
        out.append(d)
        d += dt.timedelta(days=rng.randint(18, 29))
    return out


def build_client(idx: int, seed: int) -> Client:
    rng = random.Random(f"{seed}:{idx}")
    first, last, sex = CLIENTS[idx]
    name = f"{first} {last}"
    cid = f"c{idx + 1:02d}"
    advisor = ADVISORS[idx % len(ADVISORS)]
    dates = _session_dates(rng)
    b = Builder(rng, cid, name, advisor, dates)
    n = len(dates)
    Q: list[dict] = []
    spouse = rng.choice(SPOUSES[sex]) if idx % 4 != 3 else None   # every fourth client is single

    def q(cat: str, text: str, kind: str, gold, ev_keys: list[str], traps: dict | None = None, choices=None):
        Q.append({"cat": cat, "text": text, "kind": kind, "gold": gold, "ev_keys": ev_keys,
                  "traps": {k: v for k, v in (traps or {}).items() if v is not None and v != gold}, "choices": choices})

    # spouse and family
    if spouse:
        b.at(0, lambda d: [("advisor", "Just so I have the household right, who else is part of the picture?"),
                           ("client", f"My spouse, {spouse}. We share most of the big decisions.")], "spouse")
        q("single_session", f"What is the name of {name}'s spouse?", "name", spouse, ["spouse"])
    kids0 = rng.choice([0, 0, 1, 2])
    b.at(0, lambda d: [("advisor", "And do you have children?"),
                       ("client", "Not yet." if kids0 == 0 else f"Yes, {['', 'one', 'two'][kids0]}{' kid' if kids0 == 1 else ' kids'}.")], "kids0")
    birth = rng.random() < 0.5
    kids_now = kids0
    if birth:
        s_birth = rng.randint(8, 20)
        baby = rng.choice(KIDS[rng.choice("fm")])
        kids_now = kids0 + 1
        b.at(s_birth, lambda d: [("client", f"Big news since last time: our baby, {baby}, was born last {WEEKDAYS[_last_weekday(d, rng.randint(0, 6)).weekday()]}."),
                                 ("advisor", f"Congratulations! I'll note {baby} on your file, and we should talk about beneficiaries and an RESP.")], "birth")
        q("single_session", f"What is the name of {name}'s baby born during the period of these conversations?", "name", baby, ["birth"])
    q("knowledge_update", f"How many children does {name} have now?", "int", kids_now, ["kids0"] + (["birth"] if birth else []),
      {"stale": kids0} if birth else None)

    # risk tolerance: initial, one or two changes, a reaffirmation, a hypothetical
    r0 = rng.choice([1, 2, 3])
    s1 = rng.randint(6, 11)
    r1 = r0 + rng.choice([-1, 1])
    two = rng.random() < 0.67
    s2 = rng.randint(15, 21) if two else None
    r2 = (r1 + rng.choice([-1, 1]) if 0 < r1 < 4 else (r1 + 1 if r1 == 0 else r1 - 1)) if two else None
    b.at(0, lambda d: [("advisor", "Before we set anything up, how would you describe your comfort with investment risk?"),
                       ("client", f"I'd call myself {RISK[r0]}. I can live with some ups and downs, but I don't want nasty surprises."),
                       ("advisor", f"Okay, I'll record a {RISK[r0]} risk profile on your KYC form.")], "risk0")
    reasons = ["after thinking about it over the holidays", "now that work feels more stable", "after the market drop rattled me",
               "with the family situation changing", "after reading more about long-term returns"]
    b.at(s1, lambda d: [("client", f"I want to change my risk profile, {rng.choice(reasons)}. Let's make it {RISK[r1]}."),
                        ("advisor", f"Understood. I'll update your KYC to {RISK[r1]} and rebalance the accounts to match.")], "risk1")
    if two:
        b.at(s2, lambda d: [("client", f"Can we revisit the risk setting? I'd like to move to {RISK[r2]}."),
                            ("advisor", f"Sure. Your profile will now say {RISK[r2]}, and I'll adjust the mix this week.")], "risk2")
    cur_r = r2 if two else r1
    s_re = rng.randint(s1 + 1, (s2 - 1) if two else n - 2)
    val_re = r1 if (not two or s_re < s2) else r2
    b.at(s_re, lambda d: [("advisor", f"Are you still comfortable with a {RISK[val_re]} profile?"),
                          ("client", f"Yes, {RISK[val_re]} still feels right.")], "risk_re")
    hyp = cur_r + 1 if cur_r < 4 else cur_r - 1
    s_hyp = rng.randint((s2 if two else s1) + 1, n - 1)
    b.at(s_hyp, lambda d: [("client", f"If things keep going this well, I might go {RISK[hyp]} at some point. But not now, leave it where it is."),
                           ("advisor", "Noted. We'll leave the profile as it is unless you tell me otherwise.")], "risk_hyp")
    q("knowledge_update", f"What is {name}'s current risk tolerance on file?", "label", RISK[cur_r],
      ["risk0", "risk1"] + (["risk2"] if two else []) + ["risk_hyp"],
      {"stale": RISK[r1] if two else RISK[r0], "initial": RISK[r0], "hypothetical": RISK[hyp]}, RISK)
    # the risk profile in force at the end of a month between the first and second change
    if two:
        mid = dates[rng.randint(s1, s2 - 1)]
    else:
        mid = dates[rng.randint(s1, n - 3)]
    mid_end = (mid.replace(day=28) + dt.timedelta(days=4))
    mid_end = mid_end - dt.timedelta(days=mid_end.day)
    if two and mid_end >= dates[s2]:
        mid_end = dates[s2] - dt.timedelta(days=1)
        mid_label = mid_end.isoformat()
    else:
        mid_label = f"the end of {mid_end.strftime('%B %Y')}"
    if mid_end >= dates[s1]:
        q("temporal", f"What risk tolerance was on file for {name} at {mid_label}?", "label", RISK[r1], ["risk0", "risk1"] + (["risk2"] if two else []),
          {"current": RISK[cur_r], "initial": RISK[r0]}, RISK)
    q("temporal", f"What risk tolerance did {name} start with in the first meeting?", "label", RISK[r0], ["risk0"],
      {"current": RISK[cur_r]}, RISK)

    # employer and income: a self-correction, a raise or a new job, the spouse's income, a hypothetical
    e1, e2 = rng.sample(EMPLOYERS, 2)
    role = rng.choice(ROLES)
    inc0 = rng.randrange(62_000, 118_000, 500)
    misspoke = rng.random() < 0.5
    said0 = inc0 - rng.choice([2_000, 3_000, 5_000])
    s_inc0 = rng.randint(0, 2)
    if misspoke:
        b.at(s_inc0, lambda d: [("advisor", "What does your income look like these days?"),
                                ("client", f"I work at {e1} as a {role}. I'm at {money(said0)} a year. Sorry, actually {money(inc0)} now that the bonus is part of base pay."),
                                ("advisor", f"Got it, {money(inc0)}. I'll put that on file.")], "inc0")
    else:
        b.at(s_inc0, lambda d: [("advisor", "What does your income look like these days?"),
                                ("client", f"I work at {e1} as a {role}, and my salary is {money(inc0)} a year.")], "inc0")
    job_change = rng.random() < 0.5
    s_inc1 = rng.randint(8, 18)
    inc1 = inc0 + rng.randrange(4_000, 21_000, 500)
    if job_change:
        b.at(s_inc1, lambda d: [("client", f"I have news: I accepted an offer at {e2}. The salary is {money(inc1)}."),
                                ("advisor", f"That's great. I'll update your employer to {e2} and your income to {money(inc1)}.")], "inc1")
        q("temporal", f"Where did {name} work before changing jobs?", "choice", e1, ["inc0", "inc1"], {"current": e2}, EMPLOYERS)
        q("knowledge_update", f"Who is {name}'s current employer?", "choice", e2, ["inc0", "inc1"], {"stale": e1}, EMPLOYERS)
    else:
        b.at(s_inc1, lambda d: [("client", f"I got a raise at {e1}. I'm at {money(inc1)} now."),
                                ("advisor", f"Congratulations. I'll update your income to {money(inc1)}.")], "inc1")
        q("single_session", f"Where does {name} work?", "choice", e1, ["inc0"], None, EMPLOYERS)
    sp_inc = None
    if spouse:
        sp_inc = rng.randrange(60_000, 140_000, 1_000)
        b.at(rng.randint(3, n - 2), lambda d: [("client", f"{spouse} just got promoted, so {spouse}'s salary is {money(sp_inc)} now."),
                                               ("advisor", f"Good for {spouse}. That changes the household picture a bit.")], "sp_inc")
    hyp_inc = None
    if rng.random() < 0.5:
        hyp_inc = inc1 + rng.randrange(15_000, 30_000, 1_000)
        b.at(rng.randint(s_inc1 + 1, n - 1), lambda d: [("client", f"If I get the senior role next year, I'd be at {money(hyp_inc)}. Nothing is decided yet, though."),
                                                        ("advisor", "Let's plan on the current salary and treat that as upside.")], "inc_hyp")
    q("knowledge_update", f"What is {name}'s current annual income?", "money", inc1, ["inc0", "inc1"],
      {"stale": inc0, "spouse": sp_inc, "hypothetical": hyp_inc, "misspoke": said0 if misspoke else None})
    q("temporal", f"What was {name}'s annual income before the most recent change?", "money", inc0, ["inc0", "inc1"],
      {"current": inc1, "misspoke": said0 if misspoke else None})

    # city: a move for half of the clients, and a trip that names another city
    c0, c1, c_trip = rng.sample(CITIES, 3)
    b.at(rng.randint(0, 1), lambda d: [("advisor", "Remind me where you're living these days?"),
                                       ("client", f"We're in {c0}, in a rented place for now.")], "city0")
    moved = rng.random() < 0.5
    if moved:
        b.at(rng.randint(9, 20), lambda d: [("client", f"We finally moved. We're in {c1} now."),
                                            ("advisor", f"Congratulations on the move. I'll update your address to {c1}.")], "city1")
        q("temporal", f"Where did {name} live before moving?", "choice", c0, ["city0", "city1"], {"current": c1, "trip": c_trip}, CITIES)
    b.at(rng.randint(2, n - 2), lambda d: [("client", f"We spent a long weekend in {c_trip}. Lovely, but expensive."),
                                           ("advisor", "Sounds like a nice break.")], "trip")
    q("knowledge_update", f"Which city does {name} live in now?", "choice", c1 if moved else c0, ["city0"] + (["city1"] if moved else []),
      {"stale": c0 if moved else None, "trip": c_trip}, CITIES)

    # monthly savings: advisor proposal, client decision, later change
    p_save = rng.randrange(600, 2_000, 100)
    d_save = p_save - rng.randrange(200, 500, 100)
    s_save = rng.randint(1, 4)
    b.at(s_save, lambda d: [("advisor", f"To stay on track, I'd suggest setting aside {money(p_save)} a month."),
                            ("client", f"That's a stretch right now. Let's start with {money(d_save)} a month."),
                            ("advisor", f"Okay, {money(d_save)} a month it is. I'll set up the automatic transfer.")], "save0")
    d_save2 = d_save + rng.randrange(100, 600, 50)
    b.at(rng.randint(s_save + 6, n - 3), lambda d: [("client", f"Can we bump the monthly transfer to {money(d_save2)}?"),
                                                    ("advisor", f"Done. The transfer goes to {money(d_save2)} starting next month.")], "save1")
    q("decision", f"How much per month did {name} first agree to save when the automatic transfer was set up?", "money", d_save, ["save0"],
      {"advisor_proposal": p_save, "current": d_save2})
    q("knowledge_update", f"What is {name}'s current monthly automatic savings transfer?", "money", d_save2, ["save0", "save1"],
      {"stale": d_save, "advisor_proposal": p_save})

    # contact preference
    k0 = rng.choice(CONTACT)
    b.at(0, lambda d: [("advisor", "What's the best way to reach you between meetings?"),
                       ("client", f"A {k0} is best for me.")], "contact0")
    cur_k = k0
    if rng.random() < 0.67:
        cur_k = rng.choice([k for k in CONTACT if k != k0])
        b.at(rng.randint(5, n - 2), lambda d: [("client", f"From now on, could you reach me by {cur_k} instead?"),
                                               ("advisor", f"Of course, I'll switch your contact preference to {cur_k}.")], "contact1")
    q("knowledge_update", f"How does {name} currently prefer to be contacted between meetings?", "choice", cur_k,
      ["contact0"] + (["contact1"] if cur_k != k0 else []), {"stale": k0 if cur_k != k0 else None}, CONTACT)

    # TFSA contributions across two calendar years, one reported in January about December
    tfsa: list[tuple[dt.date, int, str, int]] = []  # event date, amount, fact key, session index
    jan = [i for i, d in enumerate(dates) if d.year == 2026 and d.month == 1 and d.day >= 5]
    s_jan = jan[0] if jan else min(i for i, d in enumerate(dates) if d.year == 2026)
    amt = rng.randrange(1_500, 6_500, 500)
    dec = dt.date(2025, 12, rng.randint(27, 30))
    tfsa.append((dec, amt, f"tfsa{len(tfsa)}", s_jan))
    b.at(s_jan, (lambda a, e: lambda d: [("client", f"I forgot to mention last time: I put {money(a)} into my TFSA on December {e.day}, right before New Year's."),
                                         ("advisor", f"Thanks, I'll record the {money(a)} TFSA contribution.")])(amt, dec), tfsa[-1][2])
    pool = [i for i, d in enumerate(dates) if not (d.month == 1 and d.day < 15) and i != s_jan and i > 0]
    for s in sorted(rng.sample(pool, rng.choice([2, 3]))):
        a = rng.randrange(1_000, 7_000, 500)
        rel, days = rng.choice([("last week", 7), ("a few days ago", 3), ("yesterday", 1)])
        tfsa.append((dates[s] - dt.timedelta(days=days), a, f"tfsa{len(tfsa)}", s))
        b.at(s, (lambda a, rel: lambda d: [("client", f"I put {money(a)} into my TFSA {rel}."),
                                           ("advisor", f"Good, I'll make sure the {money(a)} is invested.")])(a, rel), tfsa[-1][2])
    total = sum(a for _, a, _, _ in tfsa)
    bal = total + rng.randrange(8_000, 25_000, 100)
    b.at(rng.randint(n - 4, n - 1), lambda d: [("advisor", f"Your TFSA is now worth about {money(bal)}, including growth."),
                                               ("client", "Nice to see it adding up.")], "tfsa_bal")
    y25 = sum(a for e, a, _, _ in tfsa if e.year == 2025)
    by_session = sum(a for _, a, _, s in tfsa if dates[s].year == 2025)  # the December top-up reported in January
    keys = [k for _, _, k, _ in tfsa]
    q("aggregation", f"In total, how much did {name} say they contributed to their TFSA during calendar year 2025?", "money", y25,
      keys, {"by_session_year": by_session, "all_years": total, "balance": bal})
    q("aggregation", f"Across all the conversations, how much has {name} said they contributed to their TFSA in total?", "money",
      total, keys, {"balance": bal, "year_2025_only": y25})

    # RRSP: advisor proposal and client decision
    s_rrsp = rng.choice([i for i, d in enumerate(dates) if d.year == 2026 and d.month in (1, 2)] or [s_jan])
    p_r = rng.randrange(5_000, 15_000, 500)
    d_r = p_r - rng.randrange(1_000, 4_000, 500)
    month = dates[s_rrsp].strftime("%B %Y")
    b.at(s_rrsp, lambda d: [("advisor", f"Before the RRSP deadline, I'd suggest contributing {money(p_r)}."),
                            ("client", f"Let's do {money(d_r)}. The rest can wait until next year."),
                            ("advisor", f"Okay, I'll process an RRSP contribution of {money(d_r)}.")], "rrsp")
    q("decision", f"How much did {name} decide to contribute to their RRSP in {month}?", "money", d_r, ["rrsp"], {"advisor_proposal": p_r})

    # goal: a home purchase with an FHSA, or a retirement age
    if rng.random() < 0.5:
        g0 = rng.randrange(650_000, 1_100_000, 25_000)
        g1 = g0 + rng.randrange(50_000, 200_000, 25_000)
        b.at(rng.randint(1, 4), lambda d: [("client", f"Our big goal is buying a place. We're aiming for something around {money(g0)}."),
                                           ("advisor", "Then the FHSA should be part of the plan. Let's look at opening one.")], "goal0")
        s_f = rng.randint(3, 8)
        wd = rng.randint(0, 6)
        f_date = _last_weekday(dates[s_f], wd)
        b.at(s_f, lambda d: [("client", f"I opened the FHSA last {WEEKDAYS[wd]}."),
                             ("advisor", "Great, I'll link it to your profile.")], "fhsa")
        b.at(rng.randint(12, n - 2), lambda d: [("client", f"Prices keep climbing, so we're now looking at around {money(g1)}."),
                                                ("advisor", f"Okay, I'll update the target to {money(g1)} in the plan.")], "goal1")
        q("knowledge_update", f"What home price is {name} currently targeting?", "money", g1, ["goal0", "goal1"], {"stale": g0})
        q("temporal", f"On what date did {name} open their FHSA?", "date", _iso(f_date), ["fhsa"], {"session_date": _iso(dates[s_f])})
        abstain_extra = [f"What interest rate did {name} lock in on their mortgage?"]
    else:
        a0 = rng.choice([58, 60, 62])
        a1 = a0 + rng.choice([2, 3, 5])
        b.at(rng.randint(1, 4), lambda d: [("client", f"I'd like to retire at {a0} if we can make it work."),
                                           ("advisor", f"Okay, I'll build the plan around retiring at {a0}.")], "goal0")
        b.at(rng.randint(12, n - 2), lambda d: [("client", f"Let's be realistic and plan for retiring at {a1} instead."),
                                                ("advisor", f"Fine, I'll move the target retirement age to {a1}.")], "goal1")
        q("knowledge_update", f"At what age does {name} currently plan to retire?", "int", a1, ["goal0", "goal1"], {"stale": a0})
        abstain_extra = [f"On what date did {name} open their FHSA?"]

    # an ethical exclusion
    sec = rng.choice(SECTORS)
    b.at(rng.randint(2, 12), lambda d: [("client", f"One request: please keep {sec} out of my portfolio."),
                                        ("advisor", f"Understood. I'll add a {sec} exclusion to your investment instructions.")], "esg")
    q("single_session", f"Which sector did {name} ask to exclude from their portfolio?", "choice", sec, ["esg"], None, SECTORS)

    # unanswerable questions: things never said
    abstain = [f"What is {name}'s credit score?", f"How much does {name} have in an employer pension plan?",
               f"Who is the employer of {name}'s spouse?" if spouse else f"What is the name of {name}'s spouse?",
               f"On what date did {name} open an RESP?"] + abstain_extra
    for text in abstain:
        q("abstention", text, "abstain", "unknown", [])

    b.render()
    client = Client(cid, name, advisor, "dev" if idx < DEV_CLIENTS else "test", _iso(dates[-1] + dt.timedelta(days=3)), b.sessions)
    for i, qq in enumerate(Q, start=1):
        qq["qid"] = f"{cid}-q{i:02d}"
        qq["cid"] = cid
        qq["split"] = client.split
        qq["evidence"] = [t for k in qq.pop("ev_keys") for t in b.evidence.get(k, [])]
        client.questions.append(qq)
    return client


def build_dataset(seed: int = 5, n_clients: int = len(CLIENTS)) -> list[Client]:
    return [build_client(i, seed) for i in range(n_clients)]


def digest(clients: list[Client]) -> str:
    blob = json.dumps([c.to_json() for c in clients], ensure_ascii=False, sort_keys=True)
    return hashlib.sha256(blob.encode("utf-8")).hexdigest()[:16]


def save(clients: list[Client], out: str | Path) -> Path:
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    with open(out / "clients.jsonl", "w", encoding="utf-8") as fh:
        for c in clients:
            fh.write(json.dumps(c.to_json(), ensure_ascii=False) + "\n")
    (out / "meta.json").write_text(json.dumps({"notice": NOTICE, "sha256_16": digest(clients), "n_clients": len(clients),
                                               "n_questions": sum(len(c.questions) for c in clients)}, indent=1), encoding="utf-8")
    return out


def load(path: str | Path) -> list[Client]:
    with open(Path(path) / "clients.jsonl", encoding="utf-8") as fh:
        return [Client.from_json(json.loads(line)) for line in fh if line.strip()]
