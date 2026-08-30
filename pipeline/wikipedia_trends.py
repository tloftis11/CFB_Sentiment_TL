"""
Wikipedia page view client.

Uses the Wikimedia "top articles" API — a single request that returns the
1,000 most-viewed articles for a given month. We then look up each team's
football article in that list and record its rank/view count.

This approach requires only 1–2 API calls regardless of how many teams we
track, entirely avoiding the per-article rate-limiting that plagues the
individual pageviews endpoint.
"""

import logging
import time
from datetime import date

import requests

logger = logging.getLogger(__name__)

WIKI_TOP_API = (
    "https://wikimedia.org/api/rest_v1/metrics/pageviews/top"
    "/en.wikipedia/all-access/{year}/{month:02d}/all-days"
)

# Maps CFBD school name → Wikipedia article title (spaces become underscores)
WIKI_TITLES: dict[str, str] = {
    "Air Force":            "Air_Force_Falcons_football",
    "Akron":                "Akron_Zips_football",
    "Alabama":              "Alabama_Crimson_Tide_football",
    "App State":            "Appalachian_State_Mountaineers_football",
    "Arizona":              "Arizona_Wildcats_football",
    "Arizona State":        "Arizona_State_Sun_Devils_football",
    "Arkansas":             "Arkansas_Razorbacks_football",
    "Arkansas State":       "Arkansas_State_Red_Wolves_football",
    "Army":                 "Army_Black_Knights_football",
    "Auburn":               "Auburn_Tigers_football",
    "Ball State":           "Ball_State_Cardinals_football",
    "Baylor":               "Baylor_Bears_football",
    "Boise State":          "Boise_State_Broncos_football",
    "Boston College":       "Boston_College_Eagles_football",
    "Bowling Green":        "Bowling_Green_Falcons_football",
    "BYU":                  "BYU_Cougars_football",
    "Buffalo":              "Buffalo_Bulls_football",
    "California":           "California_Golden_Bears_football",
    "Charlotte":            "Charlotte_49ers_football",
    "Cincinnati":           "Cincinnati_Bearcats_football",
    "Clemson":              "Clemson_Tigers_football",
    "Coastal Carolina":     "Coastal_Carolina_Chanticleers_football",
    "Colorado":             "Colorado_Buffaloes_football",
    "Colorado State":       "Colorado_State_Rams_football",
    "Duke":                 "Duke_Blue_Devils_football",
    "East Carolina":        "East_Carolina_Pirates_football",
    "Eastern Michigan":     "Eastern_Michigan_Eagles_football",
    "Florida":              "Florida_Gators_football",
    "Florida Atlantic":     "Florida_Atlantic_Owls_football",
    "Florida International":"FIU_Panthers_football",
    "Florida State":        "Florida_State_Seminoles_football",
    "Fresno State":         "Fresno_State_Bulldogs_football",
    "Georgia":              "Georgia_Bulldogs_football",
    "Georgia Southern":     "Georgia_Southern_Eagles_football",
    "Georgia State":        "Georgia_State_Panthers_football",
    "Georgia Tech":         "Georgia_Tech_Yellow_Jackets_football",
    "Hawai'i":              "Hawaii_Rainbow_Warriors_football",
    "Houston":              "Houston_Cougars_football",
    "Illinois":             "Illinois_Fighting_Illini_football",
    "Indiana":              "Indiana_Hoosiers_football",
    "Iowa":                 "Iowa_Hawkeyes_football",
    "Iowa State":           "Iowa_State_Cyclones_football",
    "James Madison":        "James_Madison_Dukes_football",
    "Kansas":               "Kansas_Jayhawks_football",
    "Kansas State":         "Kansas_State_Wildcats_football",
    "Kennesaw State":       "Kennesaw_State_Owls_football",
    "Kent State":           "Kent_State_Golden_Flashes_football",
    "Kentucky":             "Kentucky_Wildcats_football",
    "Liberty":              "Liberty_Flames_football",
    "Louisiana":            "Louisiana_Ragin'_Cajuns_football",
    "Louisiana Tech":       "Louisiana_Tech_Bulldogs_football",
    "Louisville":           "Louisville_Cardinals_football",
    "LSU":                  "LSU_Tigers_football",
    "Marshall":             "Marshall_Thundering_Herd_football",
    "Maryland":             "Maryland_Terrapins_football",
    "Memphis":              "Memphis_Tigers_football",
    "Miami":                "Miami_Hurricanes_football",
    "Miami (OH)":           "Miami_RedHawks_football",
    "Michigan":             "Michigan_Wolverines_football",
    "Michigan State":       "Michigan_State_Spartans_football",
    "Middle Tennessee":     "Middle_Tennessee_Blue_Raiders_football",
    "Minnesota":            "Minnesota_Golden_Gophers_football",
    "Mississippi State":    "Mississippi_State_Bulldogs_football",
    "Missouri":             "Missouri_Tigers_football",
    "Navy":                 "Navy_Midshipmen_football",
    "NC State":             "NC_State_Wolfpack_football",
    "Nebraska":             "Nebraska_Cornhuskers_football",
    "Nevada":               "Nevada_Wolf_Pack_football",
    "New Mexico":           "New_Mexico_Lobos_football",
    "New Mexico State":     "New_Mexico_State_Aggies_football",
    "North Carolina":       "North_Carolina_Tar_Heels_football",
    "North Texas":          "North_Texas_Mean_Green_football",
    "Northern Illinois":    "Northern_Illinois_Huskies_football",
    "Northwestern":         "Northwestern_Wildcats_football",
    "Notre Dame":           "Notre_Dame_Fighting_Irish_football",
    "Ohio":                 "Ohio_Bobcats_football",
    "Ohio State":           "Ohio_State_Buckeyes_football",
    "Oklahoma":             "Oklahoma_Sooners_football",
    "Oklahoma State":       "Oklahoma_State_Cowboys_football",
    "Ole Miss":             "Ole_Miss_Rebels_football",
    "Old Dominion":         "Old_Dominion_Monarchs_football",
    "Oregon":               "Oregon_Ducks_football",
    "Oregon State":         "Oregon_State_Beavers_football",
    "Penn State":           "Penn_State_Nittany_Lions_football",
    "Pittsburgh":           "Pittsburgh_Panthers_football",
    "Purdue":               "Purdue_Boilermakers_football",
    "Rice":                 "Rice_Owls_football",
    "Rutgers":              "Rutgers_Scarlet_Knights_football",
    "Sam Houston":          "Sam_Houston_Bearkats_football",
    "San Diego State":      "San_Diego_State_Aztecs_football",
    "San José State":       "San_José_State_Spartans_football",
    "SMU":                  "SMU_Mustangs_football",
    "South Alabama":        "South_Alabama_Jaguars_football",
    "South Carolina":       "South_Carolina_Gamecocks_football",
    "South Florida":        "South_Florida_Bulls_football",
    "Southern Miss":        "Southern_Miss_Golden_Eagles_football",
    "Stanford":             "Stanford_Cardinal_football",
    "Syracuse":             "Syracuse_Orange_football",
    "TCU":                  "TCU_Horned_Frogs_football",
    "Temple":               "Temple_Owls_football",
    "Tennessee":            "Tennessee_Volunteers_football",
    "Texas":                "Texas_Longhorns_football",
    "Texas A&M":            "Texas_A&M_Aggies_football",
    "Texas State":          "Texas_State_Bobcats_football",
    "Texas Tech":           "Texas_Tech_Red_Raiders_football",
    "Toledo":               "Toledo_Rockets_football",
    "Troy":                 "Troy_Trojans_football",
    "Tulane":               "Tulane_Green_Wave_football",
    "Tulsa":                "Tulsa_Golden_Hurricane_football",
    "UAB":                  "UAB_Blazers_football",
    "UCF":                  "UCF_Knights_football",
    "UCLA":                 "UCLA_Bruins_football",
    "UConn":                "Connecticut_Huskies_football",
    "UL Monroe":            "Louisiana_Monroe_Warhawks_football",
    "UNLV":                 "UNLV_Rebels_football",
    "USC":                  "USC_Trojans_football",
    "Utah":                 "Utah_Utes_football",
    "Utah State":           "Utah_State_Aggies_football",
    "UTEP":                 "UTEP_Miners_football",
    "UTSA":                 "UTSA_Roadrunners_football",
    "Vanderbilt":           "Vanderbilt_Commodores_football",
    "Virginia":             "Virginia_Cavaliers_football",
    "Virginia Tech":        "Virginia_Tech_Hokies_football",
    "Wake Forest":          "Wake_Forest_Demon_Deacons_football",
    "Washington":           "Washington_Huskies_football",
    "Washington State":     "Washington_State_Cougars_football",
    "West Virginia":        "West_Virginia_Mountaineers_football",
    "Western Kentucky":     "Western_Kentucky_Hilltoppers_football",
    "Western Michigan":     "Western_Michigan_Broncos_football",
    "Wisconsin":            "Wisconsin_Badgers_football",
    "Wyoming":              "Wyoming_Cowboys_football",
}

# Reverse lookup: article title → school name
_TITLE_TO_SCHOOL = {v: k for k, v in WIKI_TITLES.items()}


def _fetch_top_articles(year: int, month: int) -> dict[str, int]:
    """
    Fetch the top-1000 most-viewed Wikipedia articles for a given month.
    Returns {article_title: total_views}.
    """
    url = WIKI_TOP_API.format(year=year, month=month)
    headers = {"User-Agent": "CFBSentimentBot/1.0 (cfb analytics; open-source)"}
    try:
        resp = requests.get(url, headers=headers, timeout=15)
        if resp.status_code != 200:
            logger.warning(f"Wikipedia top-articles API: HTTP {resp.status_code}")
            return {}
        items = resp.json().get("items", [{}])[0].get("articles", [])
        return {item["article"]: item["views"] for item in items}
    except Exception as exc:
        logger.warning(f"Wikipedia top-articles API error: {exc}")
        return {}


def get_scores(schools: list[str]) -> dict[str, float]:
    """
    Return total monthly Wikipedia page views for each school's football article.

    Uses the "top articles" endpoint — 1–2 API calls total, no per-team requests.
    Teams whose articles don't appear in the top 1000 for the month get 0.0.
    """
    today = date.today()

    # Fetch current month and the previous month to cover the last ~30 days
    months_to_fetch = [(today.year, today.month)]
    prev_month = today.month - 1 or 12
    prev_year  = today.year if today.month > 1 else today.year - 1
    months_to_fetch.append((prev_year, prev_month))

    views_by_title: dict[str, int] = {}
    for year, month in months_to_fetch:
        batch = _fetch_top_articles(year, month)
        for title, views in batch.items():
            views_by_title[title] = views_by_title.get(title, 0) + views
        if len(months_to_fetch) > 1:
            time.sleep(1.0)

    # Map article views back to school names
    results: dict[str, float] = {}
    found = 0
    for school in schools:
        title = WIKI_TITLES.get(school)
        if title and title in views_by_title:
            results[school] = float(views_by_title[title])
            found += 1
        else:
            results[school] = 0.0

    logger.info(f"  Wikipedia: {found}/{len(schools)} teams found in top-1000 lists")
    return results
