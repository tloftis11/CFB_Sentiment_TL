"""
Reddit r/CFB mention client.

Fetches the top ~500 posts from r/CFB over the past month and counts
weighted engagement (upvotes + comments) for posts that mention each team.
Uses Reddit's public JSON API — no auth required.

Note: Reddit may block datacenter IPs (GitHub Actions). get_scores() returns
an empty dict on failure so the pipeline continues with the other two sources.
"""

import logging
import time
from typing import Optional

import requests

logger = logging.getLogger(__name__)

POSTS_PER_PAGE = 100
MAX_PAGES = 5          # 500 posts max — ~5 API calls, ~10s total

# Per-school search terms checked against lowercase post title.
# Terms are tuples so short ambiguous words aren't over-matched.
# The school itself is always tried as a fallback.
_TERMS: dict[str, list[str]] = {
    "Alabama":          ["alabama crimson", "alabama football", "roll tide"],
    "App State":        ["app state", "appalachian state"],
    "Arizona":          ["arizona wildcats"],
    "Arizona State":    ["arizona state", "asu football"],
    "Arkansas":         ["arkansas razorbacks", "woo pig"],
    "Army":             ["army black knights", "army football"],
    "Auburn":           ["auburn tigers", "auburn football", "war eagle"],
    "Baylor":           ["baylor bears", "baylor football"],
    "Boise State":      ["boise state"],
    "Boston College":   ["boston college"],
    "BYU":              ["byu cougars", "byu football"],
    "California":       ["cal bears", "california golden bears"],
    "Charlotte":        ["charlotte 49ers"],
    "Cincinnati":       ["cincinnati bearcats"],
    "Clemson":          ["clemson tigers", "clemson football"],
    "Coastal Carolina": ["coastal carolina"],
    "Colorado":         ["colorado buffaloes", "colorado football"],
    "Colorado State":   ["colorado state rams"],
    "Duke":             ["duke blue devils", "duke football"],
    "East Carolina":    ["east carolina", "ecu pirates"],
    "Florida":          ["florida gators", "gator football"],
    "Florida Atlantic": ["florida atlantic", "fau owls"],
    "Florida State":    ["florida state", "fsu football", "nole"],
    "Fresno State":     ["fresno state"],
    "Georgia":          ["georgia bulldogs", "uga football", "go dawgs"],
    "Georgia Southern": ["georgia southern"],
    "Georgia State":    ["georgia state"],
    "Georgia Tech":     ["georgia tech", "gt football"],
    "Houston":          ["houston cougars", "houston football"],
    "Illinois":         ["illinois football", "illini football"],
    "Indiana":          ["indiana hoosiers", "indiana football"],
    "Iowa":             ["iowa hawkeyes", "iowa football"],
    "Iowa State":       ["iowa state cyclones"],
    "Kansas":           ["kansas jayhawks", "kansas football"],
    "Kansas State":     ["kansas state", "k-state football"],
    "Kentucky":         ["kentucky wildcats", "kentucky football"],
    "Liberty":          ["liberty flames", "liberty football"],
    "Louisiana":        ["louisiana ragin cajuns", "ul football"],
    "Louisiana Tech":   ["louisiana tech"],
    "Louisville":       ["louisville cardinals", "louisville football"],
    "LSU":              ["lsu tigers", "lsu football", "geaux tigers"],
    "Marshall":         ["marshall thundering herd"],
    "Maryland":         ["maryland terrapins", "maryland football"],
    "Memphis":          ["memphis tigers", "memphis football"],
    "Miami":            ["miami hurricanes", "the u football"],
    "Miami (OH)":       ["miami redhawks"],
    "Michigan":         ["michigan wolverines", "michigan football", "go blue"],
    "Michigan State":   ["michigan state spartans", "michigan state football"],
    "Middle Tennessee": ["middle tennessee", "mtsu"],
    "Minnesota":        ["minnesota gophers", "minnesota football"],
    "Mississippi State":["mississippi state", "miss state bulldogs"],
    "Missouri":         ["missouri tigers", "mizzou football"],
    "Navy":             ["navy midshipmen", "navy football"],
    "NC State":         ["nc state wolfpack", "nc state football"],
    "Nebraska":         ["nebraska cornhuskers", "nebraska football", "huskers"],
    "Nevada":           ["nevada wolf pack", "nevada football"],
    "New Mexico":       ["new mexico lobos"],
    "New Mexico State": ["new mexico state"],
    "North Carolina":   ["unc tar heels", "north carolina tar heels"],
    "Northern Illinois":["northern illinois huskies"],
    "Northwestern":     ["northwestern wildcats", "northwestern football"],
    "Notre Dame":       ["notre dame", "fighting irish football"],
    "Ohio":             ["ohio bobcats"],
    "Ohio State":       ["ohio state buckeyes", "ohio state football", "osu football"],
    "Oklahoma":         ["oklahoma sooners", "sooner football", "boomer sooner"],
    "Oklahoma State":   ["oklahoma state cowboys", "oklahoma state football"],
    "Ole Miss":         ["ole miss rebels", "ole miss football"],
    "Old Dominion":     ["old dominion monarchs"],
    "Oregon":           ["oregon ducks", "oregon football"],
    "Oregon State":     ["oregon state beavers"],
    "Penn State":       ["penn state nittany lions", "penn state football"],
    "Pittsburgh":       ["pitt panthers", "pittsburgh football"],
    "Purdue":           ["purdue boilermakers", "purdue football"],
    "Rice":             ["rice owls", "rice football"],
    "Rutgers":          ["rutgers scarlet knights", "rutgers football"],
    "San Diego State":  ["san diego state", "sdsu football"],
    "San José State":   ["san jose state", "sjsu football"],
    "SMU":              ["smu mustangs", "smu football"],
    "South Alabama":    ["south alabama jaguars"],
    "South Carolina":   ["south carolina gamecocks", "south carolina football"],
    "South Florida":    ["south florida bulls", "usf football"],
    "Southern Miss":    ["southern miss", "usm football"],
    "Stanford":         ["stanford cardinal", "stanford football"],
    "Syracuse":         ["syracuse orange", "syracuse football"],
    "TCU":              ["tcu horned frogs", "tcu football"],
    "Tennessee":        ["tennessee volunteers", "tennessee football", "vols football"],
    "Texas":            ["texas longhorns", "hook em horns", "texas football"],
    "Texas A&M":        ["texas a&m aggies", "texas a&m football", "gig em"],
    "Texas State":      ["texas state bobcats"],
    "Texas Tech":       ["texas tech red raiders", "texas tech football"],
    "Toledo":           ["toledo rockets", "toledo football"],
    "Troy":             ["troy trojans", "troy football"],
    "Tulane":           ["tulane green wave", "tulane football"],
    "Tulsa":            ["tulsa golden hurricane"],
    "UAB":              ["uab blazers", "uab football"],
    "UCF":              ["ucf knights", "ucf football"],
    "UCLA":             ["ucla bruins", "ucla football"],
    "UConn":            ["uconn huskies", "connecticut football"],
    "UL Monroe":        ["ul monroe", "louisiana monroe"],
    "UNLV":             ["unlv rebels", "unlv football"],
    "USC":              ["usc trojans", "usc football", "fight on"],
    "Utah":             ["utah utes", "utah football"],
    "Utah State":       ["utah state aggies"],
    "UTEP":             ["utep miners"],
    "UTSA":             ["utsa roadrunners"],
    "Vanderbilt":       ["vanderbilt commodores", "vanderbilt football"],
    "Virginia":         ["virginia cavaliers", "uva football"],
    "Virginia Tech":    ["virginia tech hokies", "virginia tech football"],
    "Wake Forest":      ["wake forest demon deacons", "wake forest football"],
    "Washington":       ["washington huskies", "washington football", "go dawgs"],
    "Washington State": ["washington state cougars", "wazzu"],
    "West Virginia":    ["west virginia mountaineers", "wvu football"],
    "Western Kentucky": ["western kentucky", "wku football"],
    "Western Michigan": ["western michigan broncos"],
    "Wisconsin":        ["wisconsin badgers", "wisconsin football"],
    "Wyoming":          ["wyoming cowboys", "wyoming football"],
}


def get_scores(schools: list[str]) -> dict[str, float]:
    """
    Return a weighted engagement score for each school based on r/CFB top posts
    from the past month. Score = sum(upvotes + num_comments) across matching posts.

    Returns empty dict on failure (so callers can fall back to other sources).
    """
    posts = _fetch_posts()
    if not posts:
        return {}

    results: dict[str, float] = {s: 0.0 for s in schools}

    for post in posts:
        pdata = post.get("data", {})
        title = (pdata.get("title") or "").lower()
        score = max(int(pdata.get("score") or 0), 0)
        comments = int(pdata.get("num_comments") or 0)
        weight = score + comments

        if weight <= 0:
            continue

        for school in schools:
            if _matches(school, title):
                results[school] += weight

    matched = sum(1 for v in results.values() if v > 0)
    logger.info(f"  Reddit: scored {matched}/{len(schools)} teams from {len(posts)} posts")
    return results


def _fetch_posts() -> list[dict]:
    """Paginate r/CFB top (last month), return up to MAX_PAGES * POSTS_PER_PAGE posts."""
    session = requests.Session()
    # Reddit requires a descriptive User-Agent in this exact format per their API rules
    session.headers.update({
        "User-Agent": "script:cfb-sentiment-analyzer:1.0 (analytics project; /u/cfb-sentiment-bot)"
    })

    posts: list[dict] = []
    after: Optional[str] = None

    for page in range(MAX_PAGES):
        params: dict = {"limit": POSTS_PER_PAGE, "t": "month"}
        if after:
            params["after"] = after

        try:
            resp = session.get(
                "https://www.reddit.com/r/CFB/top.json",
                params=params,
                timeout=15,
            )
            if resp.status_code == 429:
                logger.warning("Reddit: rate limited — stopping early")
                break
            if resp.status_code != 200:
                logger.warning(f"Reddit: HTTP {resp.status_code} on page {page + 1} — aborting")
                break

            data = resp.json().get("data", {})
            children = data.get("children", [])
            posts.extend(children)
            after = data.get("after")
            if not after or not children:
                break

            if page < MAX_PAGES - 1:
                time.sleep(2.0)

        except Exception as exc:
            logger.warning(f"Reddit: error fetching page {page + 1}: {exc}")
            break

    logger.info(f"  Reddit: fetched {len(posts)} r/CFB posts")
    return posts


def _matches(school: str, title: str) -> bool:
    terms = _TERMS.get(school, [school.lower()])
    return any(t in title for t in terms)
