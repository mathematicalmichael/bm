import os
import re
import json

import hiplot
import numexpr as ne
import numpy as np
import pandas as pd
import streamlit as st
from matplotlib import pyplot as plt

# from scipy.stats import gaussian_kde as gkde

st.set_page_config(layout="wide", page_title="The Oracle", page_icon=":crystal_ball:")

hide_streamlit_style = """
<style>
.css-1y0tads {padding-top: 0rem;}
.css-r698ls {visibility: hidden;}
footer {visibility: hidden;}
#MainMenu {visibility: hidden;}
</style>
"""

st.markdown(hide_streamlit_style, unsafe_allow_html=True)

RESERVED_MATH = ["exp", "pi", "e", "tan", "cos", "sin", "sinh", "tanh", "cosh"]
RESERVED_ST = [
    "ob",
    "eq",
    "data",
]  # THIS IS A VERY IMPORTANT LINE FOR LOADING SESSION STATE FROM DISK

s = st.session_state

if "eq" not in s:
    s["eq"] = "revenue - expenses"
if "ob" not in s:
    s["ob"] = "profit"
if "data" not in st.session_state:
    s["data"] = None  # dataframe

# st.title("The Anything Calculator :mage:")
st.markdown("# The Oracle :crystal_ball:")
st.markdown("### (aka The Monte-Carlo Machine :game_die:)")
st.text(
    "This application lets you quantify the impact of uncertainty on the decisions you need to make."
)
with st.expander("Examples"):
    st.text("Copy/paste these (tap on mobile) into the equation below.")
    st.code("eth_mined_per_day * num_days_mined_per_year * dollars_per_eth")
    st.code(
        "items_sold_per_day * cost_of_item * days_worked_per_year - annual_expenses"
    )
    st.code(
        "biweekly_income * 26 - rent * 12 - weekly_food * 52 - weekly_fun * 52 - monthly_car_payment * 12 - quarterly_insurance * 4"
    )
st.markdown("---")


def clear_data():
    st.session_state["data"] = None


objective = st.text_input(
    "What are you estimating?", value=s["ob"], on_change=clear_data
)
objective = (
    str(objective).lower().replace(" ", "_").replace("-", "_").replace("/", "_per_")
)
eq = st.text_input(
    "What equation defines it?",
    value=s["eq"],
    on_change=clear_data,
)
eq = str(eq).lower().replace("^", "**")

s.eq = eq
s.ob = objective

the_input = eq.replace(" ", "")
tokens = []
variables = []

# number_or_symbol = re.compile("([^ 0-9]+)")
# zero_or_more_whitespaces, math_expr = "\s*", "[()+*/-]"
# alphanumeric, decimals, scientific = "\w+", "\.?\d*", "\d+\.?\d*"
# capture_group = f"({math_expr}|{scientific}|{decimals}|{alphanumeric})"
# magic_regex_incantation = zero_or_more_whitespaces + capture_group
# magic_regex_incantation =   # math | words | decimal | scientific
tokenizer = re.compile(r"\s*([()+*/-]|\w+|\.?\d*|\d+\.?\d*)")
# tokenizer = re.compile(r"\s*([()+*/-]|\w+|\d+)")
current_pos = 0
while current_pos < len(the_input):
    match = tokenizer.match(the_input, current_pos)
    if match is None:
        raise SyntaxError("Syntax error")
    m = match.group(1)
    # if m in ('*', '-', '^', '+', '/'):

    test = re.findall(re.compile("[a-zA-Z]+"), m)
    if test:
        variables.append(m)
    tokens.append(m)
    current_pos = match.end()


variables = set(variables) - set(RESERVED_MATH) - set(RESERVED_ST)
variables = list(variables)

for v in variables:
    if v not in s:
        s[v] = {"min": "0", "max": "1"}

for v in s:
    if v not in variables + RESERVED_ST:
        # st.write(f"deleting {v}")
        del s[v]

out = {v: s[v] for v in variables}
# st.write(out)  # PRINT DEBUGGING
# st.write(s.data)  # PRINT DEBUGGING
if out:
    # N = min(max(int(st.sidebar.text_input("Number of Samples", value=100, on_change=clear_data)), 10), 10000)
    N = st.slider(
        "Fidelity", min_value=500, max_value=25000, step=500, on_change=clear_data
    )
    st.sidebar.markdown("### Define range for each input.")
    st.sidebar.table(pd.DataFrame(out).T.astype(str).reindex(columns=['min', 'max']))

    st.sidebar.markdown("## Settings :gear:")
    x = st.sidebar.selectbox("parameter", options=variables)
    min_val = st.sidebar.text_input("minimum", value=s[x]["min"])
    max_val = st.sidebar.text_input("maximum", value=s[x]["max"])
    min_val, max_val = float(min_val), float(max_val)
    a, b = st.sidebar.columns(2)
    if st.sidebar.button("set parameter", on_click=clear_data):
        st.session_state[x]["min"] = str(min_val)
        st.session_state[x]["max"] = str(max_val)

    # TODO assert no keys missing before running.


print_eq = " ".join(tokens)
# st.text(print_eq)
# st.text(tokens)
st.markdown("### Your Model:")
st.markdown(
    f"**{objective}** = "
    + print_eq.replace(" ", "")
    .replace("**", " :arrow_up_small:")
    .replace("*", " :heavy_multiplication_x:")
    .replace("+", " :heavy_plus_sign:")
    .replace("-", " :heavy_minus_sign:")
    .replace("/", " :heavy_division_sign:")
)


if set(variables) - set(s.keys()):
    st.write("Still missing definitions. Current State:")
else:
    if out:
        simulated = st.button("run the simulation")
        if simulated:
            # st.write("Evaluating")
            D = {}
            for x in variables:
                _mx, _mn = float(s[x].get("max")), float(s[x].get("min"))
                D[x] = np.random.rand(N) * (_mx - _mn) + _mn

            predictions = ne.evaluate(eq.replace('pi', str(np.pi)), D)

            df = pd.DataFrame(D)
            df.insert(0, objective, predictions)
            df.to_csv("data.csv", index=None)
            s.data = df


if s.data is not None:
    total = s.data[objective].to_list()
    # _x = np.linspace(min(total), max(total))
    # _y = gkde(_x).pdf(_x)
    st.sidebar.markdown("## Target Predictions :8ball:")
    fig, ax = plt.subplots(1)
    try:
        ax.hist(total, bins=40)
        ax.set_xlabel(objective)
        st.sidebar.pyplot(fig)
    except ValueError:
        pass
    if st.button("visualize the result"):
        # os.system('./invoke_remote.sh data.csv > index.html')
        h = hiplot.Experiment.from_dataframe(s.data)
        h.display_st(key="hiplot")


st.sidebar.markdown("## Manage state :floppy_disk:")
fname = st.sidebar.text_input("filename", value="data")

if out is not None:
    if st.sidebar.button("save"):
        with open(f"{fname}.json", "w") as f:
            out["ob"] = objective
            out["eq"] = eq
            json.dump(out, f)
        st.sidebar.write("Saved")

if os.path.exists(f"{fname}.json"):
    if st.sidebar.button("load", on_click=clear_data):
        with open(f"{fname}.json", "r") as f:
            _out = json.load(f)
            for k in _out:
                s[k] = _out[k]
        st.sidebar.write("Loaded")
        s.data = None
        st.markdown("## state loaded:")
        st.write(s)
        st.sidebar.button("proceed")
