from flask import Flask, render_template
import pandas as pd
import matplotlib.pyplot as plt
import os
import matplotlib
matplotlib.use('Agg')  # Use non-GUI backend for Flask

app = Flask(__name__)

CSV_FILE = "s.csv"   # Your dataset
CHART_DIR = "static/charts"

@app.route("/")
def index():
    # Load data
    df = pd.read_csv(CSV_FILE)

    # === Basic Stats ===
    dept_count = df["department"].value_counts().to_dict()
    total_responses = len(df)
    total_departments = len(dept_count)

    # === LLM Usage Count ===
    llm_usage = {}
    for llm_list in df["llms"]:
        for llm in str(llm_list).split(","):
            llm = llm.strip()
            if llm:
                llm_usage[llm] = llm_usage.get(llm, 0) + 1

    total_llms = len(llm_usage)

    # === Tool Usage Count ===
    tool_usage = {}
    for tool_list in df["tools"]:
        for tool in str(tool_list).split(","):
            tool = tool.strip()
            if tool:
                tool_usage[tool] = tool_usage.get(tool, 0) + 1

    total_tools = len(tool_usage)

    # === Prompt Analysis ===
    avg_prompt_length = df["prompt"].dropna().apply(lambda x: len(str(x).split())).mean()

    # Ensure chart directory
    os.makedirs(CHART_DIR, exist_ok=True)

    # === Department Distribution Chart ===
    plt.figure(figsize=(7,4))
    df["department"].value_counts().plot(kind="bar", color="skyblue", edgecolor="black")
    plt.title("Department Distribution")
    plt.ylabel("Count")
    plt.xticks(rotation=45)
    plt.tight_layout()
    dept_chart = os.path.join(CHART_DIR, "dept.png")
    plt.savefig(dept_chart)
    plt.close()

    # === LLM Usage Chart ===
    plt.figure(figsize=(7,4))
    pd.Series(llm_usage).sort_values(ascending=False).plot(kind="bar", color="lightgreen", edgecolor="black")
    plt.title("LLM Usage Frequency")
    plt.ylabel("Count")
    plt.xticks(rotation=45)
    plt.tight_layout()
    llm_chart = os.path.join(CHART_DIR, "llm.png")
    plt.savefig(llm_chart)
    plt.close()


    return render_template(
        "index.html", 
        dept_count=dept_count, 
        llm_usage=llm_usage,
        tool_usage=tool_usage,
        dept_chart=dept_chart.replace("static/", ""),
        llm_chart=llm_chart.replace("static/", ""),
        total_responses=total_responses,
        total_departments=total_departments,
        total_llms=total_llms,
        total_tools=total_tools,
        avg_prompt_length=round(avg_prompt_length, 2)
    )

if __name__ == "__main__":
    app.run(debug=True)
