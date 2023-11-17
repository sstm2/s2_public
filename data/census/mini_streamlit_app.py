import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from s2_public.data import census
from config import settings  # this is where our API key is stored. You'll need to change this to get your own key.


@st.cache_data
def get_state_data(api_key, state_codes, college_educated=True):
    if college_educated:
        conditions = ["MAR=2,3,4,5", "SCHL=21,22,23,24"]
    else:
        conditions = ["MAR=2,3,4,5"]
    df_list = []
    for s in state_codes["STATEFP"]:
        df = census.micro_api(
            survey="acs",
            var_1="AGEP",
            var_2="SEX",
            year=2022,
            state_codes=[str(s)],
            conditions=conditions,
            api_key=api_key,
        )
        women = df.loc[2].iloc[25:41].sum()
        men = df.loc[1].iloc[25:41].sum()
        df_list.append(
            pd.DataFrame({"state": state_codes.loc[state_codes["STATEFP"] == s, "STATE"], "ratio": women / men})
        )
    return pd.concat(df_list).dropna()


def render():
    st.title("Sample Census Data Streamlit App")
    state_codes = pd.read_csv("https://www2.census.gov/geo/docs/reference/codes2020/national_state2020.txt", sep="|")
    # county_codes = pd.read_csv("https://www2.census.gov/geo/docs/reference/codes2020/national_county2020.txt", sep="|")

    college_edu = st.sidebar.radio("Education Filter", ["College Educated", "No Filter"])
    if college_edu == "College Educated":
        college_tf = True
        figure_text = "Ratio of College Educated Single Women to Men Ages 25 to 40"
    else:
        college_tf = False
        figure_text = "Ratio of College Educated Single Women to Men Ages 25 to 40"

    df = get_state_data(settings.US_CENSUS_API_KEY, state_codes, college_educated=college_tf)

    fig = go.Figure(
        data=go.Choropleth(
            locations=df["state"],  # Spatial coordinates
            z=df["ratio"].astype(float),  # Data to be color-coded
            locationmode="USA-states",  # set of locations match entries in `locations`
            colorscale="Blues",
            colorbar_title="Ratio",
        )
    )

    fig.update_layout(
        title_text=figure_text,
        geo_scope="usa",  # limite map scope to USA
    )

    st.plotly_chart(fig)


if __name__ == "__main__":
    render()
