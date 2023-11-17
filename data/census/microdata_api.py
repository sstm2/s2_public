import pandas as pd
import requests


def micro_api(
    survey="cps",
    var_1=None,
    var_2=None,
    year=2022,
    month="jan",
    state_codes=None,
    county_codes=None,
    puma_codes=None,
    conditions=None,
    weights=None,
    api_key=None,
):
    """Get CPS weights from the Census API

    Args:
        survey (str): Survey to pull from. Either "cps" or "acs".
        var_1 (str): First variable to pull from the CPS or ACS API. Defaults to age.
            See https://api.census.gov/data/2023/cps/basic/jan/variables.html and
            https://api.census.gov/data/2019/acs/acs1/pums/variables.html
        var_2 (str, optional): Second variable to pull from the CPS or ACS API.
        year (int, optional): Year to pull from.
        month (str, optional): Month to pull from. Defaults to January. Ignored for ACS.
        state_codes (list, optional): List of states to pull from. Defaults to all states.
        county_codes (list, optional): List of counties to pull from for CPS. Defaults to all counties.
        puma_codes (list, optional): List of PUMAs to pull from for ACS. Defaults to all PUMAs.
        conditions (str, optional): Conditions to filter on. For example, conditions=["PEMARITL=4,5,6"] will filter
            for never maried, seperated, or divorced.
            See documentation at https://api.census.gov/data/2023/cps/basic/jan/variables.html
        weights (str, optional): Weights to use. Defaults to individual weights.
        api_key (str, optional): API key to use. Get yours at https://api.census.gov/data/key_signup.html.
    """

    if api_key is None:
        raise ValueError("Must specify api_key; get yours here: https://api.census.gov/data/key_signup.html")
    if survey.lower() == "cps":
        micro_url = f"https://api.census.gov/data/{year}/cps/basic/{month}?"
        if weights is None:
            weights = "PWSSWGT"
    elif survey.lower() == "acs":
        micro_url = f"https://api.census.gov/data/{year}/acs/acs1/pums?"
        if weights is None:
            weights = "PWGTP"
    else:
        raise ValueError("survey must be 'cps' or 'acs'")
    if isinstance(state_codes, str):
        state_codes = [state_codes]
    micro_url += f"tabulate=weight({weights})&col+{var_1}&"
    if var_2 is not None:
        micro_url += f"row+{var_2}&"

    # if var_2 is not None:
    #     micro_url = f"https://api.census.gov/data/{year}/cps/basic/{month}?tabulate=weight({weights})&col+{var_1}&row+{var_2}&"  # {condition_string}key={settings.US_CENSUS_API_KEY}"
    # else:
    #     micro_url = f"https://api.census.gov/data/{year}/cps/basic/{month}?tabulate=weight({weights})&col+{var_1}&"  # {condition_string}key={settings.US_CENSUS_API_KEY}"
    if survey.lower() == "cps":
        if state_codes is not None:
            if county_codes is not None:
                micro_url += f"for=county:"
                for j in county_codes:
                    micro_url += f"{j},"
                if len(state_codes) > 1:
                    raise ValueError("Must specify only one state code if county_codes is specified")
                micro_url = micro_url[:-1] + f"&in=state:{state_codes[0]}&"
            else:
                micro_url += f"for=state:"
                for j in state_codes:
                    micro_url += f"{j},"
                micro_url = micro_url[:-1] + "&"
        if (county_codes is not None) and (state_codes is None):
            raise ValueError("Must specify state_codes if county_codes is specified")
    else:
        if state_codes is not None:
            if puma_codes is not None:
                micro_url += "for=public%20use%20microdata%20area:"
                for j in puma_codes:
                    micro_url += f"{j},"
                if len(state_codes) > 1:
                    raise ValueError("Must specify only one state code if puma_codes is specified")
                micro_url = micro_url[:-1] + f"&in=state:{state_codes[0]}&"
            else:
                micro_url += f"for=state:"
                for j in state_codes:
                    micro_url += f"{j},"
                micro_url = micro_url[:-1] + "&"
        if (county_codes is not None) and (state_codes is None):
            raise ValueError("Must specify state_codes if county_codes is specified")

    if conditions is not None:
        for l in conditions:
            micro_url += f"{l}&"
    micro_url += f"key={api_key}"

    print(micro_url)

    r = requests.get(micro_url)
    if r.status_code == 200:
        print("Success: 200")
    else:
        raise ValueError(f"Error: {r.status_code}", r.text)
    rj = r.json()
    cols = rj[0]
    col_list = [[k for k in v.values()][0] if isinstance(v, dict) else v for v in cols]

    df = pd.DataFrame(rj[1:], columns=col_list)
    if var_2 is not None:
        df.index = [int(j) for j in df.iloc[:, -1]]
        df = df.iloc[:, 0:-1]
        df = df.sort_index(axis=0)
    return df
