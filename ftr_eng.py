import streamlit as st
import pandas as pd
import numpy as np
import google.generativeai as genai
import io
import matplotlib.pyplot as plt
import seaborn as sns
import math
import base64
from PIL import Image
from io import BytesIO

st.set_page_config(
    page_title="Feature Engineering Assistant",
    page_icon="🧪",
    layout="wide"
)
st.title("🧪 Feature Engineering Assistant")

# Define the visualization function
def visualize_dataset(df, target):
    """
    Generate and return visualization figures for a dataset.
    
    Args:
        df: Pandas DataFrame with the dataset
        target: Target variable name
    
    Returns:
        List of matplotlib figures
    """
    sns.set(style="whitegrid")  # Style général propre et lisible
    figures = []  # List to store figure objects

    # Vérifier si la cible existe dans le DataFrame
    if target not in df.columns:
        st.error(f"Target variable '{target}' not found in the dataset.")
        return figures

    # Identifier les variables catégorielles et numériques
    cat_vars = []
    num_vars = []

    for col in df.columns:
        if col == target:
            continue
        if df[col].dtype in ['object', 'category'] or df[col].nunique() <= 10:
            cat_vars.append(col)
        else:
            num_vars.append(col)

    variables = [col for col in df.columns if col != target]
    
    # S'il n'y a pas de variables à visualiser, retourner une liste vide
    if not variables:
        st.warning("No variables to visualize.")
        return figures

    # ---------- Graphiques individuels ----------
    cols = 3
    rows = math.ceil(len(variables) / cols)

    fig1 = plt.figure(figsize=(cols*5.5, rows*4.5))
    
    for i, col in enumerate(variables):
        ax = fig1.add_subplot(rows, cols, i+1)
        if col in cat_vars:
            # Pour les variables catégorielles, utiliser countplot
            order = df[col].value_counts().sort_index().index
            palette = sns.color_palette("magma", len(order))
            sns.countplot(data=df, x=col, ax=ax, order=order, palette=palette)
            ax.set_title(f"{col} (Catégorie)", fontsize=11)
        else:
            # Pour les variables numériques, utiliser histplot
            sns.histplot(data=df, x=col, kde=True, ax=ax, color="#EAAD77")
            ax.set_title(f"{col} (Numérique)", fontsize=11)
        ax.tick_params(axis='x', rotation=45)

    plt.tight_layout()
    figures.append(fig1)

    # ---------- Relations avec la variable cible ----------
    fig2 = plt.figure(figsize=(cols*5.5, rows*4.5))
    
    for i, col in enumerate(variables):
        ax = fig2.add_subplot(rows, cols, i+1)
        if df[target].dtype in ['object', 'category'] or df[target].nunique() <= 10:
            if col in num_vars:
                sns.boxplot(x=target, y=col, data=df, ax=ax, palette="coolwarm")
            else:
                try:
                    order = df[col].value_counts().sort_index().index
                    palette = sns.color_palette("mako", len(order))
                    sns.countplot(data=df, x=col, hue=target, ax=ax, order=order, palette=palette)
                except:
                    # En cas d'erreur, essayer sans order
                    sns.countplot(data=df, x=col, hue=target, ax=ax)
        else:
            if col in num_vars:
                sns.scatterplot(x=col, y=target, data=df, ax=ax, color="teal", alpha=0.6)
            else:
                sns.boxplot(x=col, y=target, data=df, ax=ax, palette="flare")
        ax.set_title(f"{col} vs {target}", fontsize=11)
        ax.tick_params(axis='x', rotation=45)

    plt.tight_layout()
    figures.append(fig2)
    
    return figures

# Function to convert matplotlib figure to base64 encoding
def fig_to_base64(fig):
    """Convert matplotlib figure to base64 string."""
    buf = BytesIO()
    fig.savefig(buf, format="png", dpi=100)
    buf.seek(0)
    img_str = base64.b64encode(buf.read()).decode()
    return img_str

# Function to convert matplotlib figure to PIL Image
def fig_to_pil(fig):
    """Convert matplotlib figure to PIL Image."""
    buf = BytesIO()
    fig.savefig(buf, format="png", dpi=100)
    buf.seek(0)
    return Image.open(buf)

# Configure sidebar
with st.sidebar:
    st.header("Configuration")
    api_key = st.text_input("Enter Gemini API Key", type="password",value="AIzaSyCx8gyVweVrRrJfWEK371H6LMwBhZZSAiw")
    st.caption("Your API key is not stored and is only used for this session.")
    
    st.subheader("Visualization Settings")
    include_visualizations = st.checkbox("Include visualizations in prompt", value=True)
    target_variable = st.text_input("Target variable (for visualizations)", value="Fraud")

# File uploader
with st.expander("Upload Datasets", expanded=True):
    st.write("Upload your train and test datasets (CSV or Excel)")
    file = st.file_uploader(
        label="Upload Train and Test Datasets",
        type=["csv", "xlsx", "xls"],
        accept_multiple_files=True,
        help="First upload train dataset, then test dataset (optional)"
    )

# Initialize dataframes
train = pd.DataFrame()
test = pd.DataFrame()

# Descriptive stats and visualization placeholders
train_describe = ""
test_describe = ""
visualization_descriptions = ""
train_figures = []

# File reading logic
if file:
    if len(file) == 1:
        try:
            # Read single file
            train = pd.read_csv(file[0]) if file[0].type == "text/csv" else pd.read_excel(file[0])
            st.success(f"Dataset loaded: {file[0].name}")
            
            # Generate descriptive statistics
            describe_buf = io.StringIO()
            train.describe(include='all').to_string(buf=describe_buf)
            train_describe = describe_buf.getvalue()
            
            col1, col2 = st.columns(2)
            with col1:
                st.write("Dataset Preview:")
                st.dataframe(train.head(), use_container_width=True)
            with col2:
                st.write("Dataset Info:")
                buffer = io.StringIO()
                train.info(buf=buffer)
                st.text(buffer.getvalue())
            
            st.write("Descriptive Statistics:")
            st.dataframe(train.describe(include='all'), use_container_width=True)
            
            # Generate visualizations if target variable exists in dataset
            if target_variable and target_variable in train.columns and include_visualizations:
                st.write("### Dataset Visualizations")
                with st.spinner("Generating visualizations..."):
                    try:
                        train_figures = visualize_dataset(train, target_variable)
                        
                        # Display the figures in Streamlit if they were created
                        if len(train_figures) >= 1:
                            st.write("#### Individual Variable Distributions")
                            st.pyplot(train_figures[0])
                        
                        if len(train_figures) >= 2:
                            st.write(f"#### Relationships with Target ({target_variable})")
                            st.pyplot(train_figures[1])
                        
                        # Create textual descriptions of visualizations for the prompt
                        visualization_descriptions = """
                        Visualization Insights:
                        1. Individual Variable Distributions - Shows the distribution of each feature
                        2. Relationships with Target - Shows how each feature relates to the fraud target
                        """
                    except Exception as viz_error:
                        st.error(f"Error generating visualizations: {viz_error}")
                
        except Exception as e:
            st.error(f"Error reading file: {e}")

    elif len(file) >= 2:
        try:
            # Read train and test files
            train = pd.read_csv(file[0]) if file[0].type == "text/csv" else pd.read_excel(file[0])
            test = pd.read_csv(file[1]) if file[1].type == "text/csv" else pd.read_excel(file[1])

            # Generate descriptive statistics for both datasets
            train_describe_buf = io.StringIO()
            train.describe(include='all').to_string(buf=train_describe_buf)
            train_describe = train_describe_buf.getvalue()
            
            test_describe_buf = io.StringIO()
            test.describe(include='all').to_string(buf=test_describe_buf)
            test_describe = test_describe_buf.getvalue()

            st.success(f"Datasets loaded: {file[0].name} (train) and {file[1].name} (test)")
            
            tab1, tab2, tab3 = st.tabs(["Train Dataset", "Test Dataset", "Visualizations"])
            
            with tab1:
                col1, col2 = st.columns(2)
                with col1:
                    st.write("Train Dataset Preview:")
                    st.dataframe(train.head(), use_container_width=True)
                with col2:
                    st.write("Train Dataset Info:")
                    buffer = io.StringIO()
                    train.info(buf=buffer)
                    st.text(buffer.getvalue())
                st.write("Train Descriptive Statistics:")
                st.dataframe(train.describe(include='all'), use_container_width=True)
            
            with tab2:
                col1, col2 = st.columns(2)
                with col1:
                    st.write("Test Dataset Preview:")
                    st.dataframe(test.head(), use_container_width=True)
                with col2:
                    st.write("Test Dataset Info:")
                    buffer = io.StringIO()
                    test.info(buf=buffer)
                    st.text(buffer.getvalue())
                st.write("Test Descriptive Statistics:")
                st.dataframe(test.describe(include='all'), use_container_width=True)
            
            # Generate visualizations if target variable exists in dataset
            if target_variable and target_variable in train.columns and include_visualizations:
                with tab3:
                    with st.spinner("Generating visualizations..."):
                        try:
                            train_figures = visualize_dataset(train, target_variable)
                            
                            # Display the figures in Streamlit if they were created
                            if len(train_figures) >= 1:
                                st.write("#### Individual Variable Distributions")
                                st.pyplot(train_figures[0])
                            
                            if len(train_figures) >= 2:
                                st.write(f"#### Relationships with Target ({target_variable})")
                                st.pyplot(train_figures[1])
                            
                            # Create textual descriptions of visualizations for the prompt
                            visualization_descriptions = """
                            Visualization Insights:
                            1. Individual Variable Distributions - Shows the distribution of each feature
                            2. Relationships with Target - Shows how each feature relates to the fraud target
                            """
                        except Exception as viz_error:
                            st.error(f"Error generating visualizations: {viz_error}")
                    
        except Exception as e:
            st.error(f"Error reading files: {e}")

    else:
        st.warning("Please upload at least one dataset")

# Dataset description input
with st.expander("Dataset Description", expanded=True):
    desc = st.text_area(
        "Enter the description of variables in the dataset",
        value="""Overview
Objective
Detect fraudulent transactions based on transaction details, user behavior, and spending patterns using KNN.

Dataset Overview
The dataset consists of transaction data with anonymized features. Each row represents a transaction, and the goal is to classify whether it is fraudulent

Feature Description
Transaction Amount ($) – Transaction value.
Transaction Type – Categorical (Online, ATM, etc.).
Merchant Category – Categorical (Retail, Travel, etc.).
Time of Day (Hour) – Transaction time.
Day of Week – Encoded as 0 (Monday) to 6 (Sunday).
User's Past Transaction Frequency – Number of past transactions.
Average Transaction Amount (Last 30 Days) – Average spend.
User's Account Age (Months) – How long the user has had an account.
Number of Cards Linked to Account – Number of bank cards.
Transaction Location (City Level) – Encoded location.
Distance from Last Transaction (km) – Geographical distance.
Card Declined Count (Last 30 Days) – Number of declined transactions.
Fraud (Target) – 1 = Fraudulent, 0 = Legitimate.""",
        height=300,
        key="description"
    )

# Feature engineering request
with st.expander("Feature Engineering Request", expanded=True):
    order = st.text_area(
        "Specify your feature engineering requirements",
        value="Create a new database for train and test with new variables and an explanation of each new variable and display the two new databases",
        height=100,
        key="target"
    )

# Generate feature engineering plan
if st.button("Generate Feature Engineering Plan", type="primary"):
    if not api_key:
        st.error("Please enter your Gemini API key in the sidebar")
    elif train.empty:
        st.error("Please upload at least one dataset")
    else:
        try:
            with st.spinner("Generating feature engineering plan..."):
                genai.configure(api_key=api_key)
                
                # Use a more capable model for complex tasks with images
                model_name = "gemini-1.5-pro"
                st.info(f"Using {model_name} model for generating feature engineering plan.")
                model = genai.GenerativeModel(model_name)
                
                # Create prompt with context, descriptive statistics, and visualization insights
                prompt_parts = []
                
                # Add text prompt
                text_prompt = f"""
                Based on the following dataset description:
                {desc}
                
                And the following feature engineering request:
                {order}
                
                Here are the descriptive statistics of the train dataset:
                {train_describe}
                
                {"Here are the descriptive statistics of the test dataset:" + test_describe if test_describe else ""}
                
                {visualization_descriptions if include_visualizations else ""}
                
                Please provide a detailed feature engineering plan for fraud detection.
                Include specific code examples in Python using pandas and any relevant libraries.
                The plan should be specific to these datasets and leverage insights from the descriptive statistics and visualizations.
                Recommend which features should be created, transformed, or combined to improve fraud detection.
                
                For each suggested feature, explain:
                1. What is the feature
                2. Why it would be useful for fraud detection
                3. How to implement it with Python code
                
                When possible, suggest feature engineering techniques that are commonly used in fraud detection systems.
                """
                prompt_parts.append(text_prompt)
                
                # Add visualization images if available and enabled - UPDATED CODE HERE
                if include_visualizations and train_figures and len(train_figures) >= 2:
                    st.info("Including visualizations in the prompt to Gemini.")
                    for i, fig in enumerate(train_figures[:2]):  # Limit to first 2 figures to avoid token issues
                        try:
                            # Convert matplotlib figure to PIL Image
                            img = fig_to_pil(fig)
                            
                            # Use the proper method for adding image content based on API version
                            try:
                                # Add image directly to prompt_parts (new API method)
                                prompt_parts.append(img)
                            except Exception:
                                st.warning(f"Could not add image {i+1} to prompt directly. Trying alternative method...")
                                # If the above fails, try including image description instead
                                if i == 0:
                                    prompt_parts.append("First visualization shows individual variable distributions.")
                                else:
                                    prompt_parts.append(f"Second visualization shows relationships with target ({target_variable}).")
                        except Exception as img_error:
                            st.warning(f"Could not process image {i+1}: {img_error}")
                
                # Generate response
                safety_settings = {
                    genai.types.HarmCategory.HARM_CATEGORY_HATE_SPEECH: genai.types.HarmBlockThreshold.BLOCK_NONE,
                    genai.types.HarmCategory.HARM_CATEGORY_HARASSMENT: genai.types.HarmBlockThreshold.BLOCK_NONE,
                    genai.types.HarmCategory.HARM_CATEGORY_SEXUALLY_EXPLICIT: genai.types.HarmBlockThreshold.BLOCK_NONE,
                    genai.types.HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT: genai.types.HarmBlockThreshold.BLOCK_NONE,
                }
                
                generation_config = genai.types.GenerationConfig(
                    temperature=0.2,
                    top_p=0.95,
                    top_k=40,
                    max_output_tokens=8192,
                )
                
                response = model.generate_content(
                    prompt_parts,
                    safety_settings=safety_settings,
                    generation_config=generation_config
                )
                
                # Check if response has text attribute (handling different response formats)
                if hasattr(response, 'text'):
                    result = response.text
                else:
                    # Handle different response format
                    result = response.parts[0].text if hasattr(response, 'parts') else str(response)
                
                st.success("Feature engineering plan generated successfully!")
                
                # Display response in a nice format
                st.markdown("## Feature Engineering Plan")
                st.markdown(result)
                
                # Add download button
                st.download_button(
                    "Download Feature Engineering Plan",
                    result,
                    "feature_engineering_plan.txt",
                    "text/plain"
                )
                
                # Display code execution option
                st.markdown("## Execute Feature Engineering")
                if st.checkbox("Show code execution section"):
                    st.warning("⚠️ Always review generated code before executing!")
                    exec_code = st.text_area("Review and edit the code before executing", 
                                          value="# Place the generated code here to execute\n# Example:\n# train['new_feature'] = train['existing_feature'] * 2", 
                                          height=300)
                    if st.button("Execute Code"):
                        try:
                            with st.spinner("Executing code..."):
                                # Create a local scope with dataframes
                                local_vars = {"train": train, "test": test, "pd": pd, "np": np}
                                exec(exec_code, globals(), local_vars)
                                
                                # Get the modified dataframes back
                                train_modified = local_vars["train"]
                                test_modified = local_vars.get("test", pd.DataFrame())
                                
                                st.success("Code executed successfully!")
                                
                                # Show results
                                if not test_modified.empty:
                                    tab1, tab2 = st.tabs(["Modified Train Dataset", "Modified Test Dataset"])
                                    with tab1:
                                        st.dataframe(train_modified.head(), use_container_width=True)
                                    with tab2:
                                        st.dataframe(test_modified.head(), use_container_width=True)
                                else:
                                    st.dataframe(train_modified.head(), use_container_width=True)
                                
                                # Add download options for modified datasets
                                col1, col2 = st.columns(2)
                                with col1:
                                    csv = train_modified.to_csv(index=False)
                                    st.download_button(
                                        "Download Modified Train Dataset", 
                                        csv, 
                                        "modified_train.csv",
                                        "text/csv"
                                    )
                                if not test_modified.empty:
                                    with col2:
                                        csv = test_modified.to_csv(index=False)
                                        st.download_button(
                                            "Download Modified Test Dataset", 
                                            csv, 
                                            "modified_test.csv",
                                            "text/csv"
                                        )
                        except Exception as e:
                            st.error(f"Error executing code: {e}")
                
        except Exception as e:
            st.error(f"Error generating feature engineering plan: {e}")

# Footer
st.markdown("---")
st.caption("Feature Engineering Assistant powered by Google Gemini")
