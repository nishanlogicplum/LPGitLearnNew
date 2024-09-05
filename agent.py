import os
from langchain_openai import ChatOpenAI
from langchain.agents import tool
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain.agents.format_scratchpad.openai_tools import format_to_openai_tool_messages
from langchain.agents.output_parsers.openai_tools import OpenAIToolsAgentOutputParser
from langchain.agents import AgentExecutor
import ast
import markdown
from xhtml2pdf import pisa
from dotenv import load_dotenv
from git import Repo
import shutil
import errno

# Loading Keys
load_dotenv()
os.environ['OPENAI_API_KEY'] = os.getenv('OPENAI_API_KEY')
os.environ["LANGCHAIN_TRACING_V2"] = "true"
os.environ["LANGCHAIN_API_KEY"] = os.getenv('LANGCHAIN_API_KEY')

def setup_llm(model_name):
    return ChatOpenAI(model=model_name, temperature=0)

@tool
def get_python_module_names(path: str) -> list:
    """Returns the names of all .py files in the directory and its subdirectories."""
    py_modules = []
    for root, dirs, files in os.walk(path):
        for file in files:
            if file.endswith('.py'):
                py_modules.append(os.path.join(root, file))
    return py_modules

@tool
def find_folder_in_codebases(folder_name: str) -> str:
    """Finds the path of a folder located somewhere inside a folder named 'Codebases'."""
    start_path = os.getcwd()
    for root, dirs, _ in os.walk(start_path):
        if 'Codebases' in root:
            if folder_name in dirs:
                return os.path.join(root, folder_name)
    return ""

@tool
def get_folder_structure(path: str) -> str:
    """Returns the folder structure including all subdirectories, along with the names of all Python files (.py) within them."""
    def tree(dir_path: str, level: int = 0, is_last: bool = True) -> str:
        output = ""
        indent = "    " * level
        entries = os.listdir(dir_path)
        dirs = [d for d in entries if os.path.isdir(os.path.join(dir_path, d))]
        files = [f for f in entries if os.path.isfile(os.path.join(dir_path, f)) and f.endswith('.py')]
        
        for i, item in enumerate(sorted(dirs) + sorted(files)):
            is_last_item = (i == len(dirs) + len(files) - 1)
            prefix = "└── " if is_last_item else "├── "
            output += f"{indent}{prefix}{item}\n"
            if item in dirs:
                output += tree(os.path.join(dir_path, item), level + 1, is_last_item)
        return output
    return tree(path)

@tool
def read_python_file(path: str) -> str:
    """Opens a .py file, reads its content, and returns it as a string."""
    with open(path, 'r', encoding='utf-8') as file:
        return file.read()

@tool
def get_function_code(path: str, function_names: list) -> dict:
    """Extracts the code of specific functions from a .py file."""
    with open(path, 'r', encoding='utf-8') as file:
        tree = ast.parse(file.read())
    function_code = {}
    for node in ast.walk(tree):
        if isinstance(node, ast.FunctionDef) and node.name in function_names:
            function_code[node.name] = ast.get_source_segment(open(path).read(), node)
    return function_code

@tool
def find_python_file(file_name: str) -> str:
    """Finds the complete path to a .py file inside the 'Codebases' folder."""
    base_dir = 'Codebases'
    for root, dirs, files in os.walk(base_dir):
        if file_name in files:
            return os.path.join(root, file_name)
    return ''

@tool
def write_to_py_file(file_path: str, content: str) -> str:
    """Writes the given content to a .py file at the specified path."""
    try:
        os.makedirs(os.path.dirname(file_path), exist_ok=True)
        with open(file_path, 'w') as file:
            file.write(content)
        return f"Successfully wrote content to {file_path}"
    except Exception as e:
        return f"Error writing to {file_path}: {str(e)}"

@tool
def create_pdf_from_markdown(folder_name: str, markdown_content: str, file_name: str = "output.pdf") -> str:
    """Creates a PDF file in the specified folder from Markdown content."""
    try:
        os.makedirs(folder_name, exist_ok=True)
        file_path = os.path.join(folder_name, file_name)
        html = markdown.markdown(markdown_content)
        full_html = f"""
        <html>
        <head>
            <style>
                body {{ font-family: Arial, sans-serif; }}
            </style>
        </head>
        <body>
            {html}
        </body>
        </html>
        """
        with open(file_path, "wb") as pdf_file:
            pisa_status = pisa.CreatePDF(full_html, dest=pdf_file)
        if pisa_status.err:
            return f"Error creating PDF: {pisa_status.err}"
        return f"Successfully created PDF from Markdown at {file_path}"
    except Exception as e:
        return f"Error creating PDF in {folder_name}: {str(e)}"
    
import os
from difflib import SequenceMatcher

@tool
def find_most_similar_name(input_name: str) -> str:
    """
    Searches for the most similar folder or file name in the 'Codebases' directory.

    Args:
    input_name (str): The name to search for.

    Returns:
    str: The most similar folder or file name found, along with its similarity score.
    """
    base_dir = 'Codebases'
    all_names = []

    for root, dirs, files in os.walk(base_dir):
        all_names.extend(dirs)
        all_names.extend(files)

    if not all_names:
        return "No files or folders found in the Codebases directory."

    def similarity(a, b):
        return SequenceMatcher(None, a.lower(), b.lower()).ratio()

    most_similar = max(all_names, key=lambda name: similarity(input_name, name))
    similarity_score = similarity(input_name, most_similar)

    return f"Most similar name: '{most_similar}' (Similarity: {similarity_score:.2f})"

def setup_agent(model_name):
    llm = setup_llm(model_name)
    tools = [get_python_module_names, get_folder_structure, find_folder_in_codebases, 
             read_python_file, get_function_code, find_python_file, write_to_py_file, 
             create_pdf_from_markdown, find_most_similar_name]
    
    prompt = ChatPromptTemplate.from_messages([
        ("system", "You are very powerful assistant, but don't know current events"),
        ("user", "{input}"),
        MessagesPlaceholder(variable_name="agent_scratchpad"),
    ])

    llm_with_tools = llm.bind_tools(tools)

    agent = (
        {
            "input": lambda x: x["input"],
            "agent_scratchpad": lambda x: format_to_openai_tool_messages(x["intermediate_steps"]),
        }
        | prompt
        | llm_with_tools
        | OpenAIToolsAgentOutputParser()
    )

    return AgentExecutor(agent=agent, tools=tools, verbose=True)

def clone(project_name, repo_url):
    """
    Clones a git repository into a specified directory.

    Parameters:
    project_name (str): The name of the project. This will be the name of the directory where the repo will be cloned.
    repo_url (str): The URL of the git repository to clone.

    Returns:
    None
    """

    # Construct the full path where the repo will be cloned
    repo_path = os.path.join('Codebases', project_name)

    # Check if the directory already exists
    if not os.path.exists(repo_path):
        # If not, create a new directory
        os.mkdir(repo_path)

        # Clone the repository into the newly created directory
        Repo.clone_from(repo_url, to_path=repo_path)

def local_clone(project_name, source_folder):
    """
    Copies a folder from a specified location into the 'Codebases' directory 
    with the specified project name.

    Parameters:
    project_name (str): The name of the project. This will be the name of the directory where the folder will be copied.
    source_folder (str): The path of the folder to copy.

    Returns:
    None
    """

    # Use absolute paths
    source_folder = os.path.abspath(source_folder)
    destination_path = os.path.abspath(os.path.join('Codebases', project_name))

    # Handle long path names (for Windows)
    if os.name == 'nt':
        source_folder = '\\\\?\\' + source_folder
        destination_path = '\\\\?\\' + destination_path

    # Check if the directory already exists
    if not os.path.exists(destination_path):
        try:
            # Copy the folder to the new directory
            shutil.copytree(source_folder, destination_path)
        except OSError as exc:
            if exc.errno == errno.ENOENT:
                print(f"Error: Source directory does not exist: {source_folder}")
            elif exc.errno == errno.EEXIST:
                print(f"Error: Destination directory already exists: {destination_path}")
            else:
                print(f"Error occurred while copying directory: {exc}")
    else:
        print(f"Destination directory already exists: {destination_path}")



def invoke(messages, model_name):
    agent_executor = setup_agent(model_name)
    
    # Construct the input for the agent from the chat history
    chat_history = "\n".join([f"{m['role']}: {m['content']}" for m in messages])
    query = f"Chat history:\n{chat_history}\n\nPlease respond to the latest user message."

    try:
        result = list(agent_executor.stream({"input": query}))
        return result[-1]['output']
    except:
        result = list(agent_executor.stream({"input": "Repeat this sentence without rephrasing : Sorry, I dont understand the folder or file path you are referring to. Can you be more specific with the path?"})) 
        return result[-1]['output']