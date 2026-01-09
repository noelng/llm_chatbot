import os
import json
import random
from typing import List, Dict, Any
import google.generativeai as genai
from sentence_transformers import SentenceTransformer
import chromadb
from chromadb.config import Settings


class VectorStore:
    """Handle vector database operations"""
    
    def __init__(self, persist_directory: str, embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2"):
        self.embedding_model = SentenceTransformer(embedding_model)
        self.client = chromadb.PersistentClient(path=persist_directory)
        self.collection = self.client.get_or_create_collection(name="reviews")
    
    def search(self, query: str, k: int = 10) -> List[Dict[str, Any]]:
        """Search for similar documents"""
        query_embedding = self.embedding_model.encode(query).tolist()
        
        results = self.collection.query(
            query_embeddings=[query_embedding],
            n_results=k
        )
        
        documents = []
        if results['documents'] and results['documents'][0]:
            for doc, metadata in zip(results['documents'][0], results['metadatas'][0]):
                documents.append({
                    'content': doc,
                    'metadata': metadata
                })
        
        return documents


class ReviewsTool:
    """Tool for answering questions about patient reviews"""
    
    def __init__(self, vector_store: VectorStore, llm):
        self.vector_store = vector_store
        self.llm = llm
        self.name = "Reviews"
        self.description = """Useful when you need to answer questions about patient reviews or experiences at the hospital. 
Not useful for answering questions about specific visit details such as payer, billing, treatment, diagnosis, 
chief complaint, hospital, or physician information. Pass the entire question as input."""
    
    def run(self, query: str) -> str:
        """Execute the reviews tool"""
        # Retrieve relevant documents
        documents = self.vector_store.search(query, k=10)
        
        # Format context
        context = "\n\n".join([doc['content'] for doc in documents])
        
        # Create prompt
        prompt = f"""Your job is to use patient reviews to answer questions about their experience at a hospital. 
Use the following context to answer questions. Be as detailed as possible, but don't make up 
any information that's not from the context. If you don't know an answer, say you don't know.

Context:
{context}

Question: {query}

Answer:"""
        
        # Generate response
        response = self.llm.generate_content(prompt)
        return response.text


class WaitTimeTool:
    """Tool for getting current wait times at hospitals"""
    
    def __init__(self):
        self.name = "Waits"
        self.description = """Use when asked about current wait times at a specific hospital. 
This tool ONLY accepts the hospital name, like "A", "B", or "C". 
For example: If the question is "What is the wait time at hospital C?", input should be "C". 
Do NOT include the word "hospital" or any other words, only the single letter name."""
    
    def run(self, hospital_name: str) -> int:
        """Get current wait time for a hospital (simulated)"""
        # Clean the input
        hospital_name = hospital_name.strip().upper()
        
        # Remove "hospital" if present
        hospital_name = hospital_name.replace("HOSPITAL", "").strip()
        
        # Simulate wait times
        wait_times = {
            'A': random.randint(10, 60),
            'B': random.randint(10, 60),
            'C': random.randint(10, 60)
        }
        
        return wait_times.get(hospital_name, "Unknown hospital")


class Agent:
    """Simple ReAct-style agent"""
    
    def __init__(self, llm, tools: List[Any], max_iterations: int = 5):
        self.llm = llm
        self.tools = {tool.name: tool for tool in tools}
        self.max_iterations = max_iterations
    
    def run(self, query: str, verbose: bool = True) -> Dict[str, Any]:
        """Execute the agent"""
        conversation_history = []
        intermediate_steps = []
        
        # Create system prompt
        tool_descriptions = "\n".join([
            f"- {name}: {tool.description}" 
            for name, tool in self.tools.items()
        ])
        
        system_prompt = f"""You are a helpful assistant that can use tools to answer questions.

Available tools:
{tool_descriptions}

To use a tool, respond in this format:
Thought: [your reasoning]
Action: [tool name]
Action Input: [input for the tool]

When you have the final answer, respond in this format:
Thought: I now have enough information to answer
Final Answer: [your answer]

Begin!"""
        
        for iteration in range(self.max_iterations):
            # Build prompt
            if iteration == 0:
                prompt = f"{system_prompt}\n\nQuestion: {query}\n\nThought:"
            else:
                prompt = f"{system_prompt}\n\nQuestion: {query}\n\n"
                for step in intermediate_steps:
                    prompt += f"Thought: {step['thought']}\n"
                    prompt += f"Action: {step['action']}\n"
                    prompt += f"Action Input: {step['action_input']}\n"
                    prompt += f"Observation: {step['observation']}\n\n"
                prompt += "Thought:"
            
            # Generate response
            response = self.llm.generate_content(prompt)
            response_text = response.text
            
            if verbose:
                print(f"\n{'='*50}")
                print(f"Iteration {iteration + 1}")
                print(f"{'='*50}")
                print(response_text)
            
            # Parse response
            if "Final Answer:" in response_text:
                final_answer = response_text.split("Final Answer:")[-1].strip()
                return {
                    'input': query,
                    'output': final_answer,
                    'intermediate_steps': intermediate_steps
                }
            
            # Extract action and action input
            try:
                thought = response_text.split("Action:")[0].strip()
                action_line = response_text.split("Action:")[1].split("Action Input:")[0].strip()
                action_input_line = response_text.split("Action Input:")[1].strip()
                
                # Clean up action input
                action_input = action_input_line.replace('"', '').replace("'", '').strip()
                
                # Execute tool
                if action_line in self.tools:
                    observation = self.tools[action_line].run(action_input)
                    
                    intermediate_steps.append({
                        'thought': thought,
                        'action': action_line,
                        'action_input': action_input,
                        'observation': observation
                    })
                    
                    if verbose:
                        print(f"\nObservation: {observation}")
                else:
                    if verbose:
                        print(f"\nError: Unknown tool '{action_line}'")
                    break
                    
            except Exception as e:
                if verbose:
                    print(f"\nError parsing response: {e}")
                break
        
        return {
            'input': query,
            'output': "Could not determine final answer",
            'intermediate_steps': intermediate_steps
        }


def main():
    # Load environment variables
    from dotenv import load_dotenv
    load_dotenv()
    
    # Initialize Google Gemini
    genai.configure(api_key=os.getenv('GOOGLE_API_KEY'))
    llm = genai.GenerativeModel('gemini-2.0-flash-exp')
    
    # Initialize vector store
    review_chroma_path = r"C:\Users\noeln\Desktop\llm\20260106_llm_chatbot\langchain_intro\chromadb\chroma_data"
    vector_store = VectorStore(persist_directory=review_chroma_path)
    
    # Initialize tools
    reviews_tool = ReviewsTool(vector_store, llm)
    wait_time_tool = WaitTimeTool()
    
    # Initialize agent
    agent = Agent(llm, tools=[reviews_tool, wait_time_tool])
    
    # Test queries
    print("\n" + "="*80)
    print("QUERY 1: Current wait time at hospital C")
    print("="*80)
    result1 = agent.run("What is the current wait time at hospital C?", verbose=True)
    print(f"\nFinal Answer: {result1['output']}")
    
    print("\n\n" + "="*80)
    print("QUERY 2: Patient comfort reviews")
    print("="*80)
    result2 = agent.run("What have patients said about their comfort at the hospital?", verbose=True)
    print(f"\nFinal Answer: {result2['output']}")


if __name__ == "__main__":
    main()
