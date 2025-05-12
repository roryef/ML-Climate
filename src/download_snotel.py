import os
import requests
from config import RAW_SNOTEL_DIR

# SNOTEL station IDs
STATIONS = {
    "1030": "Arapaho_Ridge",
    "322": "Bear_Lake",
    "1061": "Bear_River",
    "327": "Beartown",
    "1041": "Beaver_Ck_Village",
    "335": "Berthoud_Summit",
    "345": "Bison_Lake",
    "1185": "Black_Mesa",
    "1161": "Black_Mountain",
    "369": "Brumley",
    "938": "Buckskin_Joe",
    "913": "Buffalo_Park",
    "378": "Burro_Mountain",
    "380": "Butte",
    "387": "Cascade_2",
    "1326": "Castle_Peak",
    "1101": "Chapman_Tunnel",
    "1059": "Cochetopa_Pass",
    "408": "Columbine",
    "409": "Columbine_Pass",
    "904": "Columbus_Basin",
    "412": "Copeland_Lake",
    "415": "Copper_Mountain",
    "426": "Crosho",
    "430": "Culebra_2",
    "431": "Cumbres_Trestle",
    "438": "Deadman_Hill",
    "457": "Dry_Lake",
    "936": "Echo_Lake",
    "465": "El_Diente_Peak",
    "467": "Elk_River",
    "1252": "Elkhead_Divide",
    "1120": "Elliot_Ridge",
    "1325": "Elwood_Pass",
    "1186": "Fool_Creek",
    "485": "Fremont_Pass",
    "1057": "Glen_Cove",
    "1058": "Grayback",
    "505": "Grizzly_Peak",
    "1102": "Hayden_Pass",
    "1187": "High_Lonesome",
    "531": "Hoosier_Pass",
    "1122": "Hourglass_Lake",
    "538": "Idarado",
    "542": "Independence_Pass",
    "547": "Ivanhoe",
    "935": "Jackwhacker_Gulch",
    "551": "Joe_Wright",
    "970": "Jones_Pass",
    "556": "Kiln",
    "564": "Lake_Eldora",
    "565": "Lake_Irene",
    "580": "Lily_Pond",
    "586": "Lizard_Head_Pass",
    "589": "Lone_Cone",
    "1123": "Long_Draw_Resv",
    "940": "Lost_Dog",
    "602": "Loveland_Basin",
    "607": "Lynx_Pass",
    "905": "Mancos",
    "618": "Mc_Clure_Pass",
    "1040": "Mccoy_Park",
    "914": "Medano_Pass",
    "622": "Mesa_Lakes",
    "937": "Michigan_Creek",
    "624": "Middle_Creek",
    "1014": "Middle_Fork_Camp",
    "629": "Mineral_Creek",
    "632": "Molas_Lake",
    "1124": "Moon_Pass",
    "658": "Nast_Lake",
    "1031": "Never_Summer",
    "663": "Niwot",
    "669": "North_Lost_Trail",
    "675": "Overland_Res",
    "680": "Park_Cone",
    "682": "Park_Reservoir",
    "688": "Phantom_Valley",
    "701": "Porphyry_Creek",
    "709": "Rabbit_Ears",
    "1324": "Rat_Creek",
    "1032": "Rawah",
    "713": "Red_Mountain_Pass",
    "717": "Ripple_Creek",
    "718": "Roach",
    "939": "Rough_And_Tumble",
    "1100": "Saint_Elmo",
    "1128": "Sargents_Mesa",
    "1251": "Sawtooth",
    "737": "Schofield_Pass",
    "739": "Scotch_Creek",
    "1060": "Sharkstooth",
    "762": "Slumgullion",
    "773": "South_Colony",
    "780": "Spud_Mountain",
    "793": "Stillwater_Creek",
    "797": "Stump_Lakes",
    "802": "Summit_Ranch",
    "825": "Tower",
    "827": "Trapper_Lake",
    "829": "Trinchera",
    "838": "University_Camp",
    "839": "Upper_Rio_Grande",
    "840": "Upper_San_Juan",
    "1141": "Upper_Taylor",
    "1005": "Ute_Creek",
    "842": "Vail_Mountain",
    "843": "Vallecito",
    "1188": "Wager_Gulch",
    "1160": "Weminuche_Creek",
    "857": "Whiskey_Ck",
    "1042": "Wild_Basin",
    "869": "Willow_Creek_Pass"
}


# SNOTEL data elements
ELEMENTS = "WTEQ::value,PREC::value,TMAX::value,TMIN::value,TAVG::value,PRCP::value"

# Constructs NRCS Report Generator URL that requests full Period of Record data
def build_url(station_id):
    return (
        f"https://wcc.sc.egov.usda.gov/reportGenerator/view_csv/customSingleStationReport/"
        f"daily/{station_id}:CO:SNTL%7Cid=%22%22%7Cname/POR_BEGIN,POR_END/{ELEMENTS}"
    )

# Downloads CSV for a single SNOTEL station
def download_station_csv(station_id, name):
    url = build_url(station_id)
    os.makedirs(RAW_SNOTEL_DIR, exist_ok=True)

    filename = f"{station_id}_{name}.csv"
    filepath = os.path.join(RAW_SNOTEL_DIR, filename)

    print(f"Downloading data for {name} ({station_id})...")
    try:
        response = requests.get(url, timeout=15)
    except requests.exceptions.RequestException as e:
        print(f"Network error for {station_id}: {e}")
        return

    if response.status_code == 200:
        text = response.text
        if "Date," in text:
            with open(filepath, "w") as f:
                f.write(text)
            print(f"Saved to {filepath}")
        else:
            print(f"⚠️  No usable tabular data returned for {name} ({station_id}) — skipping.")
            print(text[:250])  # Preview warning
    else:
        print(f"Failed for {station_id}. Status: {response.status_code}")

# Downloads data for all SNOTEL stations
def download_all_snotel_stations():
    for station_id, name in STATIONS.items():
        download_station_csv(station_id, name)

if __name__ == "__main__":
    download_all_snotel_stations()
