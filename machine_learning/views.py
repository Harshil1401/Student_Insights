from django.shortcuts import render, redirect
from django.contrib import messages
import joblib
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import io
import base64
from django.contrib.auth.decorators import login_required

# Create your views here.
def dashboard_view(request):
    user_id = request.session.get('user_id')
    if not user_id:
        messages.error(request, "You must be logged in to view the dashboard.")
        return redirect('login')

    user_name = request.session.get('user_name', 'User')
    return render(request, "dashboard.html", {'user_name': user_name})

def result_view(request):
    context = {
        'prediction': request.session.get('prediction_result', 'No prediction found.'),
        'accuracy': request.session.get('model_accuracy', 'N/A'),
        'graph_data': request.session.get('graph_data', None)
    }
    return render(request, 'result.html', context)

# --- ML Model Loading and Logic ---
MODEL_ACCURACY = 0
placed_averages = {}

try:
    
    model = joblib.load('machine_learning/ml/placement_model_reduced.pkl')
    df = pd.read_csv('machine_learning/ml/Placement_Data_Full_Class.csv')

    X = df.drop(['sl_no', 'salary', 'status', 'etest_p', 'mba_p', 'specialisation'], axis=1, errors='ignore')
    y = df['status'].apply(lambda x: 1 if x == 'Placed' else 0)
    MODEL_ACCURACY = model.score(X, y)
    print(f"Model loaded with accuracy: {MODEL_ACCURACY:.2%}")
    placed_df = df[df['status'] == 'Placed']
    comparison_cols = ['ssc_p', 'hsc_p', 'degree_p']
    averages = placed_df[comparison_cols].mean().to_dict()
    placed_averages = averages
except FileNotFoundError:
    print("FATAL ERROR: Ensure 'placement_model_reduced.pkl' and 'Placement_Data_Full_Class.csv' are in the same directory as views.py.")

def generate_comparison_graph(user_scores, average_scores):
    """Generates a glassmorphism-style grouped bar chart comparing user scores to averages."""
    labels = {
        'ssc_p': '10th %',
        'hsc_p': '12th %',
        'degree_p': 'Degree %',
    }
    
    categories = list(labels.values())
    user_values = [user_scores[key] for key in labels.keys()]
    avg_values = [average_scores[key] for key in labels.keys()]

    x = np.arange(len(categories))
    width = 0.35

    fig, ax = plt.subplots(figsize=(10, 6), dpi=150)

    # Transparent figure & axis background
    fig.patch.set_alpha(0.0)
    ax.set_facecolor("none")

    # Dark/glassmorphism colors
    rects1 = ax.bar(
        x - width/2, user_values, width, label='Your Score',
        color='#4A90E2', alpha=0.8, edgecolor='white', linewidth=0.8
    )
    rects2 = ax.bar(
        x + width/2, avg_values, width, label='Placed Average',
        color='#50E3C2', alpha=0.8, edgecolor='white', linewidth=0.8
    )

    # Labels and title in white
    ax.set_ylabel('Percentage Scores', size=12, color='white')
    ax.set_title('Your Scores vs. Average of Placed Students', size=16, pad=20, color='white')
    ax.set_xticks(x)
    ax.set_xticklabels(categories, color='white', fontsize=11)
    ax.set_ylim(0, 110)

    # Minimal soft grid
    ax.yaxis.grid(True, color='white', alpha=0.05)
    ax.xaxis.grid(False)

    # Borders
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_color('white')
    ax.spines['bottom'].set_color('white')

    # Data labels in white
    ax.bar_label(rects1, padding=3, fmt='%.1f', color='white', fontsize=9)
    ax.bar_label(rects2, padding=3, fmt='%.1f', color='white', fontsize=9)

    # Legend
    legend = ax.legend(loc='upper center', bbox_to_anchor=(0.5, -0.1), ncol=2, frameon=False)
    for text in legend.get_texts():
        text.set_color("white")

    fig.tight_layout(pad=2)

    # Save with transparency
    buf = io.BytesIO()
    fig.savefig(buf, format='png', transparent=True)
    buf.seek(0)
    graph_data = base64.b64encode(buf.read()).decode('utf-8')
    plt.close(fig)
    return graph_data

def placement_predictor_view(request):
    
    user_id = request.session.get('user_id')
    if not user_id:
        messages.error(request, "You must be logged in to view the dashboard.")
        return redirect('login')
    
    if request.method == 'POST':
            user_scores = {
                'ssc_p': float(request.POST.get('ssc_p')),
                'hsc_p': float(request.POST.get('hsc_p')),
                'degree_p': float(request.POST.get('degree_p')),
            }
            input_features = {
                'gender': request.POST.get('gender'),
                'ssc_b': request.POST.get('ssc_b'),
                'hsc_b': request.POST.get('hsc_b'),
                'hsc_s': request.POST.get('hsc_s'),
                'degree_t': request.POST.get('degree_t'),
                'workex': request.POST.get('workex'),
            }
            model_input_dict = {**user_scores, **input_features}
            input_df = pd.DataFrame({k: [v] for k, v in model_input_dict.items()})
            prediction_code = model.predict(input_df)[0]

            request.session['prediction_result'] = "High Chance of Placement! Your profile is strong." if prediction_code == 1 else "Further Improvement Recommended. Focus on enhancing key skills and projects."
            request.session['model_accuracy'] = f"{MODEL_ACCURACY:.2%}"
            if placed_averages:
                request.session['graph_data'] = generate_comparison_graph(user_scores, placed_averages)
            
            return redirect('result_page')
    return render(request, 'placement_predictor.html')


# Load model and dataset
model_data = joblib.load("machine_learning/ml/ctc_predictor.pkl")
pipeline = model_data["model"]
metrics = model_data.get("metrics", {})
model_accuracy = metrics.get("r2_score", None)

dataset = pd.read_csv("machine_learning/ml/synthetic_placement_dataset.csv")  # <-- Path to your CSV

def generate_graph(branch, avg_gpa, prediction, hackathons):
    """Generates glassmorphism-style comparison graph between user input and branch average"""
    # Get branch averages
    branch_data = dataset[dataset["Branch"] == branch]
    if branch_data.empty:
        avg_values = dataset[["Average GPA", "CTC (LPA)", "Hackathons Participated"]].mean().to_dict()
    else:
        avg_values = {
            "Average GPA": branch_data["Average GPA"].mean(),
            "CTC (LPA)": branch_data["CTC (LPA)"].mean(),
            "Hackathons Participated": branch_data["Hackathons Participated"].mean()
        }

    # Prepare user values
    user_values = {
        "Average GPA": avg_gpa,
        "CTC (LPA)": prediction,
        "Hackathons Participated": hackathons
    }

    categories = list(user_values.keys())
    user_scores = list(user_values.values())
    avg_scores = [avg_values[cat] for cat in categories]

    x = np.arange(len(categories))
    width = 0.35

    fig, ax = plt.subplots(figsize=(10, 6), dpi=150)

    # Transparent figure background
    fig.patch.set_alpha(0.0)
    ax.set_facecolor("none")

    # Bar colors matching dark/glassmorphism gradient
    rects1 = ax.bar(x - width/2, user_scores, width, 
                    label='Your Input', 
                    color='#4A90E2', alpha=0.8, edgecolor='white', linewidth=0.8)
    rects2 = ax.bar(x + width/2, avg_scores, width, 
                    label='Branch Average', 
                    color='#50E3C2', alpha=0.8, edgecolor='white', linewidth=0.8)

    # Labels and titles in white
    ax.set_ylabel('Values', size=12, color='white')
    ax.set_title('Your Inputs vs Branch Averages', size=16, pad=20, color='white')
    ax.set_xticks(x)
    ax.set_xticklabels(categories, color='white', fontsize=11)
    ax.set_ylim(0, max(max(user_scores), max(avg_scores)) * 1.2)

    # Minimalist subtle grid
    ax.yaxis.grid(True, color='white', alpha=0.05)
    ax.xaxis.grid(False)

    # Remove top/right borders, make bottom/left subtle
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_color('white')
    ax.spines['bottom'].set_color('white')

    # Labels on bars
    ax.bar_label(rects1, padding=3, fmt='%.2f', color='white', fontsize=9)
    ax.bar_label(rects2, padding=3, fmt='%.2f', color='white', fontsize=9)

    # Legend styling
    legend = ax.legend(loc='upper center', bbox_to_anchor=(0.5, -0.1), ncol=2, frameon=False)
    for text in legend.get_texts():
        text.set_color("white")

    fig.tight_layout(pad=2)

    # Save with transparent background
    buf = io.BytesIO()
    fig.savefig(buf, format='png', transparent=True)
    buf.seek(0)
    graph_data = base64.b64encode(buf.read()).decode('utf-8')
    plt.close(fig)

    return graph_data


def ctc_predictor(request):

    user_id = request.session.get('user_id')
    if not user_id:
        messages.error(request, "You must be logged in to view the dashboard.")
        return redirect('login')

    if request.method == "POST":
        # Get user inputs
        gender = request.POST.get("gender")
        branch = request.POST.get("branch")
        avg_gpa = float(request.POST.get("avg_gpa", 0))
        backlogs = int(request.POST.get("backlogs", 0))
        attendance = float(request.POST.get("attendance", 0))

        skills = ",".join(request.POST.getlist("skills"))
        certifications = ",".join(request.POST.getlist("certifications"))
        internship_domain = ",".join(request.POST.getlist("internship_domain"))
        job_role = ",".join(request.POST.getlist("job_role"))
        company_tier = request.POST.get("company_tier")

        hackathons = int(request.POST.get("hackathons", 0))
        interview_score = int(request.POST.get("interview_score", 0))
        soft_skills = int(request.POST.get("soft_skills", 0))
        offer_type = request.POST.get("offer_type")

        english_score = 7  # default

        # Features for prediction
        feature_columns = [
            'Gender', 'Branch', 'Average GPA', 'Backlogs', 'Attendance (%)',
            'Skills', 'Certifications', 'Internship Domain', 'Job Role',
            'Company Tier', 'English Proficiency Score', 'Hackathons Participated',
            'Interview Performance (1-10)', 'Soft Skills Rating (1-10)', 'Offer Type'
        ]

        X = pd.DataFrame([[gender, branch, avg_gpa, backlogs, attendance,
                           skills, certifications, internship_domain, job_role,
                           company_tier, english_score, hackathons,
                           interview_score, soft_skills, offer_type]],
                         columns=feature_columns)

        # Make prediction
        pre = pipeline.predict(X)[0]
        accuracy_display = f"{model_accuracy * 100:.2f}%" if model_accuracy else "N/A"
        

        company_tier = request.POST.get("company_tier")
        if(company_tier=="Tier 2"):
            pre = pre-10
        elif(company_tier=="Tier 3"):
            pre = pre/10
        else:
            pre = pre
        
        # Generate graph inside same request
        graph_data = generate_graph(branch, avg_gpa, pre, hackathons)

        prediction = str(round(pre,2)) + " LPA"

        return render(request, "result.html", {
            "prediction": prediction,
            "accuracy": accuracy_display,
            "graph_data": graph_data
        })

    return render(request, "ctc_predictor.html")


# Load the model once at startup
Sem8_gpa_model = joblib.load("machine_learning/ml/Sem8_gpa_forecaster.pkl")
Sem7_gpa_model = joblib.load("machine_learning/ml/Sem7_gpa_forecaster.pkl")
Sem6_gpa_model = joblib.load("machine_learning/ml/Sem6_gpa_forecaster.pkl")
Sem5_gpa_model = joblib.load("machine_learning/ml/Sem5_gpa_forecaster.pkl")
Sem4_gpa_model = joblib.load("machine_learning/ml/Sem4_gpa_forecaster.pkl")
Sem3_gpa_model = joblib.load("machine_learning/ml/Sem3_gpa_forecaster.pkl")
Sem2_gpa_model = joblib.load("machine_learning/ml/Sem2_gpa_forecaster.pkl")

MODEL_MAP = {
    2: Sem2_gpa_model,
    3: Sem3_gpa_model,
    4: Sem4_gpa_model,
    5: Sem5_gpa_model,
    6: Sem6_gpa_model,
    7: Sem7_gpa_model,
    8: Sem8_gpa_model,
}

df_gpa = pd.read_csv("machine_learning/ml/synthetic_full_gpa_data.csv")

def generate_gpa_comparison_graph(user_gpas, predicted_gpa, target_semester):
    # Prepare semester labels
    semesters = [f"Sem{i}" for i in range(1, target_semester + 1)]

    # Dataset averages
    avg_gpas = []
    for i in range(1, target_semester + 1):
        col_name = f"Sem{i} GPA"  # match your CSV columns
        if i < target_semester:
            avg_gpas.append(df_gpa[col_name].mean())
        else:
            avg_gpas.append(predicted_gpa)

    # User GPA list
    user_values = user_gpas + [predicted_gpa]

    x = np.arange(len(semesters))
    width = 0.35

    fig, ax = plt.subplots(figsize=(10, 6), dpi=150)

    # Transparent background
    fig.patch.set_alpha(0.0)
    ax.set_facecolor("none")

    # Bars
    rects1 = ax.bar(
        x - width/2, user_values, width, label='Your GPA',
        color='#4A90E2', alpha=0.8, edgecolor='white', linewidth=0.8
    )
    rects2 = ax.bar(
        x + width/2, avg_gpas, width, label='Dataset Avg GPA',
        color='#50E3C2', alpha=0.8, edgecolor='white', linewidth=0.8
    )

    # Labels & styling
    ax.set_ylabel('GPA', size=12, color='white')
    ax.set_title(f'Your GPA vs. Dataset Average (Up to Sem {target_semester})',
                 size=16, pad=20, color='white')
    ax.set_xticks(x)
    ax.set_xticklabels(semesters, color='white', fontsize=11)
    ax.set_ylim(0, 10)  # GPA scale

    # Minimal grid
    ax.yaxis.grid(True, color='white', alpha=0.05)
    ax.xaxis.grid(False)

    # Borders
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_color('white')
    ax.spines['bottom'].set_color('white')

    # Data labels
    ax.bar_label(rects1, padding=3, fmt='%.2f', color='white', fontsize=9)
    ax.bar_label(rects2, padding=3, fmt='%.2f', color='white', fontsize=9)

    # Legend
    legend = ax.legend(loc='upper center', bbox_to_anchor=(0.5, -0.1),
                       ncol=2, frameon=False)
    for text in legend.get_texts():
        text.set_color("white")

    fig.tight_layout(pad=2)

    # Save as base64 PNG
    buf = io.BytesIO()
    fig.savefig(buf, format='png', transparent=True)
    buf.seek(0)
    graph_data = base64.b64encode(buf.read()).decode('utf-8')
    plt.close(fig)

    return graph_data

def gpa_forecast_view(request):

    user_id = request.session.get('user_id')
    if not user_id:
        messages.error(request, "You must be logged in to view the dashboard.")
        return redirect('login')

    prediction = None
    accuracy = None
    graph_data = None

    if request.method == "POST":

        target_semester = int(request.POST.get("target_semester", 0))
        features = [float(request.POST.get(f"sem{i}_gpa", 0)) for i in range(1, target_semester)]

        model_data = MODEL_MAP.get(target_semester)
        if not model_data:
            return render(request, "gpa_forecast.html", {"error": "Invalid semester selected"})

        model = model_data["model"]
        prediction = model.predict(np.array(features).reshape(1, -1))[0]
        metrics = model_data["metrics"]
        accuracy = round(metrics["r2_score"] * 100, 2)

        # Call separate graph generator
        graph_data = generate_gpa_comparison_graph(features, prediction, target_semester)

        return render(request, "result.html", {
            "prediction": round(prediction, 2),
            "accuracy": accuracy,
            "graph_data": graph_data
        })

    return render(request, "gpa_forecast.html")
