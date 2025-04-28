import streamlit as st
import pandas as pd
from langchain_openai import ChatOpenAI
from langgraph.graph import StateGraph,MessagesState,START,END
from typing_extensions import Annotated,TypedDict
from langchain_core.messages import SystemMessage,HumanMessage,ToolMessage
import plotly.express as px
from cryptography.fernet import Fernet
import re

encrypted_key=b'gAAAAABoCkZy6AcjIj_MpvrcyzdHVpYgmG9gLTJxjtNVDtklx6pUGJs9G3asZ1uzc8JgIXNbaPtqyF05TDvYiJVtxktpZLBmGOJTUK1eiZ46fHw70Nq94eEq5797e-VePP-iFTw6sd752z8oClPpzmT1V3uneNdtPfUx1WbfTAQOaElFggdVHeMfW4nNe2xXeJtm4XVW0cfu99n2lFTs9izH7ODOkx7KgvFLc1VUENupXLg4PSNgOn_bCPTTOSIdKFGQGos98ICXjsx6wPYQpY0yoXRd1sM8OYsH0nYTo3A0Fc7YMnChuyY='

key=b'dNxmct5HrAjsG8LsJACGnXElHLVcbzjOgsJgvdfsUck='

fernet = Fernet(key)
decr_api = fernet.decrypt(encrypted_key).decode()

st.title('Data Visualization Agent📊')

uploaded_file = st.file_uploader("Upload a data file")
if uploaded_file is not None:
    dataframe = pd.read_csv(uploaded_file)
    st.dataframe(dataframe)
    dataframe.to_csv(uploaded_file.name,index=False)

user_prompt=st.text_input('Enter your Prompt')
if st.button("Ask"):
    llm = ChatOpenAI(model='gpt-4.1-mini',temperature=0.0,api_key=decr_api)

    def code_executor_function(code_to_be_executed,repl_variables):
        sanitized_code=re.sub(r"^(\s|`)*(?i:python)?\s*", "",code_to_be_executed)
        sanitized_code=re.sub(r"(\s|`)*$", "",sanitized_code)
        exec(sanitized_code,repl_variables)
        return repl_variables

    class mainstate(TypedDict):
        dataset:pd.DataFrame
        llm_generated_code:str
        dataset_metadata:str
        task_plan:str
        user_prompt:str
        llm_code_execution_result:dict

    def metadata_generator(state):
        df=state.get('dataset')
        df_columns=df.columns.to_list()
        df_column_dtypes={k:'string' if v=='object' else v for k,v in (df.dtypes.to_dict()).items()}
        sample_col_data=[]
        for col_name in df_columns:
            single_col_data={col_name:(df.loc[0:2,col_name]).to_list()}
            sample_col_data.append(single_col_data)
        
        dataset_metadata=f'''
                            Columns in Dataset:{df_columns}\n\n
                            Column Datatypes:{df_column_dtypes}\n\n
                            Sample Data in each Column:{sample_col_data}\n\n'''
        
        return {'dataset_metadata':dataset_metadata}

    def planner_node(state):

        print('\nPlanner Execution started........\n')
        planner_llm_system_prompt=f'''**Role:** You are a highly specialized AI Planning Agent. Your sole purpose is to meticulously analyze a user's question about a dataset and determine if it can be answered with a plot, then decompose it into a sequence of clear, actionable, and logically ordered tasks *to create that plot*.

        **Target Audience for Tasks:** These tasks are **NOT** for you to execute. They are instructions for a *separate* AI agent (the "Code Execution Agent") which has access to a Python environment with Pandas and plotting libraries like Plotly, and can execute generated code against a loaded DataFrame (typically named `df`).

        **Input You Will Receive:**
        1.  'USER_QUESTION': The natural language question asked by the user about their data.
        2.  'DATA_CONTEXT': Information about the dataset the user is asking about. This typically includes:
            *   Filename (e.g., `library_data.csv`)
            *   Column names (e.g., `['Book_ID', 'Title', 'Author', 'Genre', 'Status', 'Condition', 'Checkout_Date', 'Times_Borrowed']`)
            *   Data types (e.g., `'Book_ID': 'object', 'Times_Borrowed': 'int64', ...`)
            *   (Optional) A brief description or head/sample rows of the data.

        **Your Primary Task:**
        First, determine if the user's question "{state.get('user_prompt')}" can be effectively answered by generating a plot using the provided data.
        *   If **YES**, generate a JSON formatted list of strings. Each string represents a single, discrete task that the Code Execution Agent needs to perform using Python (Pandas, Plotly) to ultimately *generate the required plot*.
        *   If **NO**, your output must be *only* the exact string: `Cannot be answered through a plot`

        **Guiding Principles for Task Creation (if plotting is possible):**
        1.  **Plot Suitability First:** Ensure the question asks for a relationship, comparison, distribution, or composition that is well-suited for visualization (e.g., bar chart, line chart, scatter plot, histogram). If the question asks for a specific value, a list, or a textual description, it's likely *not* suitable for a plot as the primary answer format.
        2.  **Decomposition:** Break down the process into the smallest logical steps. This includes data preparation (filtering, grouping, sorting, aggregation) *specifically needed for the plot*, followed by plot creation steps.
        3.  **Clarity & Precision:** Tasks must be unambiguous. Use precise column names. State exactly what calculation, manipulation, or plot configuration is needed (e.g., "Calculate the sum of 'Times_Borrowed' grouped by 'Genre'", "Filter rows where 'Status' is 'Available'", "Set 'Genre' as the x-axis", "Set the calculated sum as the y-axis").
        4.  **Logical Sequence:** The order of tasks must reflect the workflow: Data selection/preparation -> Plot definition -> Plot finalization/storage. If any required column or data is not available, stop and output `Cannot be answered through a plot`.
        5.  **Data Prep & Plot Focus:** Frame tasks in terms of DataFrame operations needed to prepare data for plotting *and* defining the plot itself (selecting plot type like bar, line, scatter; specifying x/y axes, titles, labels, colors if necessary). Assume the Code Execution Agent uses Plotly.
        6.  **State Management (Implicit):** Assume the Code Execution Agent maintains the state of the DataFrame (`df`) and intermediate data structures between tasks.
        7.  **Final Result:** The final task(s) should clearly instruct the Code Execution Agent to create the plot object (e.g., using `plotly.express`) and store it in a designated variable (e.g., `final_output_fig`) for later use/display.

        **CRITICAL Constraints - What NOT To Do:**
        *   **DO NOT** attempt to answer the user's question yourself or generate the plot image/data.
        *   **DO NOT** generate *any* Python code. Your output is *only* the list of tasks *or* the specific "Cannot..." string.
        *   **DO NOT** perform calculations or data analysis. Your job is planning the steps, not executing them.
        *   **DO NOT** output anything other than the JSON list of task strings *OR* the exact string `Cannot be answered through a plot`. No conversational text, no explanations outside the tasks (unless a task *is* explicitly about describing something needed for the plot).
        *   **DO NOT** make assumptions if data is missing or unsuitable for plotting; default to `Cannot be answered through a plot`.

        **Output Format:**
        *   **If plotting is possible:** Strictly output a JSON formatted list of strings. `["Task 1", "Task 2", ...]`
        *   **If plotting is not possible/suitable:** Strictly output *only* the following string: `Cannot be answered through a plot`

        **Example (Plotting Possible):**
        If the user's question is "Show the number of books per genre?" and provides relevant metadata:

        Your Output (Example Only):
        [
        "Group the DataFrame 'df' by the 'Genre' column.",
        "Calculate the count of entries (e.g., count 'Book_ID') for each genre.",
        "Reset the index of the grouped data to make 'Genre' and the count regular columns.",
        "Rename the count column to a descriptive name like 'Number_of_Books'.",
        "Create a bar plot using Plotly Express.",
        "Set the 'Genre' column as the x-axis.",
        "Set the 'Number_of_Books' column as the y-axis.",
        "Set the plot title to 'Number of Books per Genre'.",
        "Store the resulting Plotly figure object."
        ]

        **Example (Plotting Not Possible):**
        If the user's question is "What is the Book_ID of the book titled 'The Little Engine'?"

        Your Output (Example Only):
        Cannot be answered through a plot

        Please find the Metadata of the dataset below
        {state.get('dataset_metadata')}'''

        plan=(llm.invoke([SystemMessage(content=planner_llm_system_prompt)])).content
        print('Task Plan:',plan)

        return {'task_plan':plan}

    def code_gen_node(state):
        print('\n\nCode Generation started......\n')

        code_gen_llm_system_prompt_template=f"""**Role:** You are an expert Python Code Generation Agent specializing in using the Pandas library for data manipulation and the Plotly Express library for data visualization.

        **Primary Goal:** Your sole purpose is to translate a given sequence of data preparation and plotting tasks into a single, executable block of Python code. This code will use Pandas (`pd`) for data handling and Plotly Express (`px`) to generate a plot object based on the plan. This code will be executed using a provided Python REPL tool.

        **Input You Will Receive:**
        1.  `TASK_PLAN`: A list of natural language strings, where each string describes a specific step (data manipulation or plot configuration) to be performed. This plan was generated by a planning agent.
        2.  `DATA_CONTEXT`: Metadata about the target dataset, including:
            *   Filename
            *   Column names
            *   Column data types
            *   Sample data rows (potentially)

        **Execution Environment Context:**
        *   You **MUST** assume that a Pandas DataFrame object is already loaded and available in the execution environment under the variable name `df`.
        *   You **MUST** assume that the Pandas library is already imported and available under the alias `pd`.
        *   You **MUST** assume that the Plotly Express library is already imported and available under the alias `px`.

        **Your Task:**
        1.  Carefully analyze the `TASK_PLAN` provided below.
        2.  Translate the *entire sequence* of tasks into a single, coherent block of Python code using Pandas (`pd`) for any necessary data preparation and Plotly Express (`px`) for generating the plot. The code must operate on the DataFrame `df`.
        3.  Ensure the generated code directly addresses each step in the `TASK_PLAN` in the correct order. Data preparation steps should precede plotting steps.
        4.  The final step in your generated code **MUST** be the creation of a Plotly figure object using a `px` function (e.g., `px.bar`, `px.line`, `px.scatter`).
        5.  This final Plotly figure object **MUST** be assigned to a variable named exactly `final_output_fig`.
        6.  Your *only* output should be the generated Python code block as a single string.
        7.  If in the Task plan if anything is mentioned that graphs cannot be created for the user prompt then politely respond to the user that it is not possible to create a graph and do not tell anything else.

        **CRITICAL Constraints - What NOT To Do:**
        *   **DO NOT** output *anything* other than the Python code block string. No explanations, no conversational text, no markdown formatting like ```python ... ```, no introductory phrases (e.g., "Here is the code:").
        *   **DO NOT** include `import pandas as pd` in your code block. Assume `pd` is already available.
        *   **DO NOT** include `import plotly.express as px` in your code block. Assume `px` is already available.
        *   **DO NOT** include code to load the CSV file (e.g., `pd.read_csv(...)`). Assume `df` is already loaded.
        *   **DO NOT** wrap your code in a function definition (`def ...:`).
        *   **DO NOT** include any code to display the plot (e.g., `final_output_fig.show()`). Only generate the code that *creates* the figure object and assigns it to `final_output_fig`.
        *   **DO NOT** invent columns or use columns not mentioned in the `DATA_CONTEXT`. If a task seems impossible with the given columns, generate code that attempts the task as described, relying on the execution environment to potentially raise an error.

        **Inputs for this specific request are given below:**
        **TASK_PLAN:**
        {state.get('task_plan')}

        **DATA_CONTEXT:**
        {state.get('dataset_metadata')}

        **Example OUTPUT Format which you must always follow is given below (Example assumes TASK_PLAN was for plotting top 5 borrowed books):**
        top_books = df.sort_values(by='Times_Borrowed', ascending=False).head(5)
        final_output_fig = px.bar(top_books, x='Title', y='Times_Borrowed', title='Top 5 Most Borrowed Books', labels={{'Times_Borrowed': 'Number of Times Borrowed', 'Title': 'Book Title'}}, text='Times_Borrowed')
        final_output_fig.update_traces(textposition='outside')
        final_output_fig.update_layout(xaxis_tickangle=-30)
        """
        llm_generated_code=(llm.invoke([SystemMessage(content=code_gen_llm_system_prompt_template)])).content
        print('LLM Generated Code:\n',llm_generated_code)
        return {'llm_generated_code':llm_generated_code}


    def code_exe_node(state):
        print('\n\nCode Execution Started......\n')
        code_to_be_executed=state.get('llm_generated_code')
        df=state.get('dataset')
        repl_variables={"df":df,"pd":pd,"px":px}
        llm_code_execution_result=code_executor_function(code_to_be_executed,repl_variables)
        print('Repl_Variables:',llm_code_execution_result.keys())
        return {'llm_code_execution_result':llm_code_execution_result}

    builder=StateGraph(mainstate)
    builder.add_node('Metadata Generator',metadata_generator)
    builder.add_node('Planner',planner_node)
    builder.add_node('Code Generator',code_gen_node)
    builder.add_node('Code Executor',code_exe_node)

    builder.add_edge(START,'Metadata Generator')
    builder.add_edge('Metadata Generator','Planner')
    builder.add_edge('Planner','Code Generator')
    builder.add_edge('Code Generator','Code Executor')
    builder.add_edge('Code Executor',END)

    with st.spinner(text="Analyzing your data"):
        graph=builder.compile()

        graph_result=graph.invoke({'user_prompt':user_prompt,'dataset':dataframe})

    st.plotly_chart(graph_result['llm_code_execution_result']['final_output_fig'])