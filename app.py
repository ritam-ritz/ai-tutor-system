"""
Agentic AI Tutor System using LangGraph
Run this in Google Colab

Install dependencies first:
!pip install langgraph langchain google-generativeai gradio
"""
import os
import json
import re
from typing import TypedDict, Literal
from langgraph.graph import StateGraph, END
import google.generativeai as genai
import gradio as gr

# =======================
# CONFIGURATION
# =======================
# Replace with your Gemini API key from: https://aistudio.google.com/app/apikey
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
if not GEMINI_API_KEY:
    raise ValueError("⚠️ GEMINI_API_KEY environment variable not set!")

MODEL_NAME = "gemini-2.5-flash"
genai.configure(api_key=GEMINI_API_KEY)

# Configure Gemini directly

model = genai.GenerativeModel(MODEL_NAME)

def call_gemini(prompt, system_instruction="You are an expert educational assessment designer."):
    """Call Gemini API safely (SDK-compatible)"""
    try:
        full_prompt = f"""SYSTEM ROLE:
{system_instruction}

USER TASK:
{prompt}
"""
        response = model.generate_content(full_prompt)
        return response.text
    except Exception as e:
        raise Exception(f"Gemini API Error: {str(e)}")


# =======================
# AGENT STATE
# =======================
class AgentState(TypedDict):
    """State that the agent maintains and updates autonomously"""
    topic: str
    student_model: dict
    learning_goal: str
    current_quiz: dict
    quiz_history: list
    student_answers: list
    weak_concepts: list
    error_concepts: list  # NEW: Track concepts with errors in last quiz
    attempts: int
    overall_mastery: float
    agent_logs: list
    next_action: str
    session_complete: bool
    awaiting_user_input: bool
    user_message: str
    last_quiz_had_mistakes: bool

# =======================
# UTILITY FUNCTIONS
# =======================

def normalize_concept(concept: str) -> str:
    """Normalize concept names to ensure consistency across quizzes"""
    # Remove spaces, hyphens, underscores, and convert to lowercase
    normalized = re.sub(r'[\s\-_]+', '', concept.lower())
    # Remove special characters except alphanumeric
    normalized = re.sub(r'[^a-z0-9]', '', normalized)
    return normalized

def extract_json(text):
    """Extract JSON from LLM response"""
    # Try to find JSON block
    match = re.search(r'\{[\s\S]*\}', text)
    if match:
        try:
            return json.loads(match.group())
        except json.JSONDecodeError:
            return None
    return None

def log_action(state: AgentState, action: str, details: str):
    """Log agent's autonomous actions"""
    state["agent_logs"].append({
        "action": action,
        "details": details,
        "timestamp": f"Step {len(state['agent_logs']) + 1}"
    })
    print(f"[AGENT LOG] {action}: {details}")  # Debug print
    return state

# =======================
# AGENT DECISION NODE
# =======================
def agent_decision_node(state: AgentState) -> AgentState:
    """AUTONOMOUS DECISION MAKING"""
    print("[DECISION NODE] Entering decision node")
    log_action(state, "THINKING", "Analyzing student performance and deciding next action")
    
    mastery = state["overall_mastery"]
    attempts = state["attempts"]
  
    if state.get("last_quiz_had_mistakes") and state["error_concepts"]:
        state["next_action"] = "teach"
        state["last_quiz_had_mistakes"] = False  # reset after scheduling teaching
        log_action(state, "DECISION", f"Mistakes detected in: {', '.join(state['error_concepts'])}. Triggering teaching before next quiz.")
        return state
    
    # Goal: Achieve 80%+ mastery
    if mastery >= 0.8:
        state["next_action"] = "celebrate"
        log_action(state, "DECISION", f"Mastery achieved ({mastery:.0%})! Ending session.")
        state["session_complete"] = True
        
    elif attempts == 0:
        state["next_action"] = "diagnose"
        log_action(state, "DECISION", "No assessment yet. Starting with diagnostic quiz.")
        
    elif mastery < 0.5 and attempts > 0:
        state["next_action"] = "teach"
        log_action(state, "DECISION", f"Low mastery ({mastery:.0%}). Switching to teaching mode.")
        
    elif attempts > 2 and mastery < 0.7:
        state["next_action"] = "teach"
        log_action(state, "DECISION", f"Attempts: {attempts}, Mastery: {mastery:.0%}. Need instruction.")
        
    else:
        state["next_action"] = "adaptive_quiz"
        log_action(state, "DECISION", f"Mastery: {mastery:.0%}. Continuing adaptive assessment.")
    
    print(f"[DECISION NODE] Next action: {state['next_action']}")
    return state

# =======================
# DIAGNOSTIC NODE
# =======================
def diagnostic_node(state: AgentState) -> AgentState:
    """Generate initial diagnostic quiz with 6 questions"""
    print("[DIAGNOSTIC NODE] Generating diagnostic quiz")
    log_action(state, "GENERATING", "Creating diagnostic assessment")
    
    prompt = f"""First, write a short beginner-friendly introduction (80–120 words) to "{state['topic']}" 
to help the student get familiar with the topic.

Then create a 6-question multiple-choice diagnostic quiz on "{state['topic']}".


Difficulty level: BEGINNER to INTERMEDIATE.

Goal: Help the student LEARN the topic while answering.

Guidelines:
- Questions must be clear, simple, and concept-focused
- Avoid heavy academic language
- Test understanding of core ideas, not memorization
- Each question should teach something important about {state['topic']}
- Prefer "why" and "how it works" in an intuitive way
- Use real learning-oriented scenarios when possible

You MUST respond with ONLY valid JSON in this exact format:
{{
 "intro":"short friendly introduction to the topic",
 "questions":[
   {{"id":1, "question":"Clear beginner-friendly conceptual question", "options":["Correct conceptual idea","Common misconception","Partially correct idea","Irrelevant idea"], "answer":"A", "concept":"core-concept-1"}},
   {{"id":2, "question":"Simple intermediate question about how something works", "options":["Option A","Option B","Option C","Option D"], "answer":"B", "concept":"core-concept-2"}},
   {{"id":3, "question":"Question that connects two ideas in {state['topic']}", "options":["Option A","Option B","Option C","Option D"], "answer":"C", "concept":"core-concept-3"}},
   {{"id":4, "question":"Practical conceptual question that improves understanding", "options":["Option A","Option B","Option C","Option D"], "answer":"D", "concept":"core-concept-4"}},
   {{"id":5, "question":"Theoretical question (containing formulas if applicable)", "options":["Option A","Option B","Option C","Option D"], "answer":"A", "concept":"core-concept-5"}},
   {{"id":6, "question":"Calculation based problem", "options":["Option A","Option B","Option C","Option D"], "answer":"B", "concept":"core-concept-6"}}
 ]
}}

CRITICAL:
- Exactly 6 questions
- Beginner–intermediate level only
- Questions must directly help someone learn {state['topic']}
- If topic is programming → focus on logic, behavior, and reasoning
- If topic is science → focus on intuition, cause–effect, and meaning
- No advanced math, no research-level theory
- No explanations, ONLY JSON."""

    
    try:
        response = call_gemini(
            prompt,
            system_instruction="You are an expert educational assessment designer. Create questions that help students learn the specified topic."
        )
        
        print(f"[GEMINI RESPONSE] {response[:200]}...")
        quiz = extract_json(response)
        
        if quiz and "questions" in quiz:
            state["current_quiz"] = quiz
            state["awaiting_user_input"] = True
            state["user_message"] = f"📘 INTRODUCTION:\n\n{quiz.get('intro','')}\n\n---\nNow answer these questions:"
            log_action(state, "QUIZ_READY", f"Diagnostic quiz with {len(quiz['questions'])} questions ready")
        else:
            raise Exception("Invalid quiz format from Gemini")
            
    except Exception as e:
        print(f"[ERROR] {str(e)}")
        raise Exception(f"Failed to generate quiz. Please check your Gemini API key and try again. Error: {str(e)}")
    
    return state

# =======================
# TEACHING NODE
# =======================
def teaching_node(state: AgentState) -> AgentState:
    """Generate micro-lesson for error concepts only"""
    print("[TEACHING NODE] Creating lesson")
    # Use error_concepts if available, otherwise fall back to weak_concepts
    concepts = state['error_concepts'][:2] if state['error_concepts'] else state['weak_concepts'][:2]
    if not concepts:
        concepts = ["fundamentals"]
    
    log_action(state, "TEACHING", f"Creating lesson for: {', '.join(concepts)}")
    
    prompt = f"""Create a brief theoretical lesson on these concepts in {state['topic']}: {', '.join(concepts)}

The lesson must be specifically about {state['topic']} and help the student understand these concepts deeply.

Include:
1. Clear theoretical explanation (2-3 sentences) specific to {state['topic']}
2. One concrete example from {state['topic']}
3. One key insight or principle to remember about {state['topic']}

Keep it under 150 words and focused entirely on teaching {state['topic']}.

CRITICAL: Content must be directly relevant to learning {state['topic']}.
If {state['topic']} is Physics, teach physics concepts.
If {state['topic']} is Programming, teach programming concepts.
If {state['topic']} is Biology, teach biological concepts."""
    
    try:
        response = call_gemini(
            prompt,
            system_instruction=f"You are an expert tutor teaching {state['topic']}. Create focused, topic-specific lessons."
        )
        lesson = response
    except Exception as e:
        print(f"[ERROR] {str(e)}")
        raise Exception(f"Failed to generate lesson. Error: {str(e)}")
    
    log_action(state, "LESSON_DELIVERED", f"Taught: {', '.join(concepts)}")
    state["next_action"] = "adaptive_quiz"
    state["user_message"] = f"📚 LESSON:\n\n{lesson}\n\n---\nNow let's practice!"
    
    return state

# =======================
# ADAPTIVE QUIZ NODE
# =======================
def adaptive_quiz_node(state: AgentState) -> AgentState:
    """Generate adaptive quiz targeting ONLY concepts with errors in previous quiz"""
    print("[ADAPTIVE QUIZ NODE] Creating adaptive quiz")
    log_action(state, "ADAPTING", "Creating targeted adaptive quiz based on previous errors")
    
    # Use error_concepts (concepts where mistakes were made in last quiz)
    target_concepts = state["error_concepts"][:4] if state["error_concepts"] else ["general"]
    
    prompt = f"""First, write a short refresher (50–80 words) explaining these concepts in "{state['topic']}" where the student made mistakes:
{', '.join(target_concepts)}

Then create a 4-question adaptive quiz on "{state['topic']}" focusing ONLY on these concepts where errors occurred:

Difficulty: BEGINNER to INTERMEDIATE.

Goal: Strengthen understanding of the specific concepts where mistakes were made.

Guidelines:
- Questions must be simple, clear, and learning-focused
- Each question should address the specific errors from the previous quiz
- Avoid abstract or research-level theory
- Prefer conceptual + practical understanding
- Language should feel like a good teacher, not a textbook

ONLY return valid JSON:
{{
 "intro":"short refresher explaining these concepts where mistakes were made",
 "questions":[
   {{"id":1, "question":"Clear beginner-friendly question about {target_concepts[0]} in {state['topic']}", "options":["Correct understanding","Common mistake","Confusing idea","Wrong idea"], "answer":"A", "concept":"{target_concepts[0]}"}},
   {{"id":2, "question":"Intermediate question about how {target_concepts[1] if len(target_concepts) > 1 else target_concepts[0]} works", "options":["Option A","Option B","Option C","Option D"], "answer":"B", "concept":"{target_concepts[1] if len(target_concepts) > 1 else target_concepts[0]}"}},
   {{"id":3, "question":"Practical conceptual question that improves understanding of {target_concepts[2] if len(target_concepts) > 2 else target_concepts[0]}", "options":["Option A","Option B","Option C","Option D"], "answer":"C", "concept":"{target_concepts[2] if len(target_concepts) > 2 else target_concepts[0]}"}},
   {{"id":4, "question":"Application-based question that tests deeper understanding of {target_concepts[3] if len(target_concepts) > 3 else target_concepts[0]}", "options":["Option A","Option B","Option C","Option D"], "answer":"D", "concept":"{target_concepts[3] if len(target_concepts) > 3 else target_concepts[0]}"}}
 ]
}}

CRITICAL:
- Exactly 4 questions
- Focus ONLY on concepts where the student made errors in the previous quiz
- Beginner–intermediate only
- Must help the student learn {state['topic']}
- No heavy theory, no jargon
- Focus on understanding, not memorizing
- No explanations, ONLY JSON."""

    
    try:
        response = call_gemini(
            prompt,
            system_instruction="You are an expert educational assessment designer. Create questions that help students correct their specific mistakes and deeply understand the topic."
        )
        quiz = extract_json(response)
        
        if quiz and "questions" in quiz:
            state["current_quiz"] = quiz
            refresher = quiz.get("intro", "").strip()

            if refresher:
                state["user_message"] = f"🧠 **QUICK REFRESHER:**  \n{refresher}  \n\n---\n\nTry these questions:"
            else:
                state["user_message"] = "Try these questions:"
        else:
            raise Exception("Invalid quiz format from Gemini")
            
    except Exception as e:
        print(f"[ERROR] {str(e)}")
        raise Exception(f"Failed to generate adaptive quiz. Error: {str(e)}")
    
    state["awaiting_user_input"] = True
    log_action(state, "ADAPTIVE_QUIZ_READY", f"Generated questions targeting: {', '.join(target_concepts)}")
    
    return state

# =======================
# EVALUATION NODE
# =======================
def evaluation_node(state: AgentState) -> AgentState:
    """Evaluate student answers and update student model"""
    print("[EVALUATION NODE] Evaluating answers")
    log_action(state, "EVALUATING", "Analyzing student responses")
    
    quiz = state["current_quiz"]
    answers = state["student_answers"]
    
    concept_stats = {}
    correct = 0
    mistakes = 0
    error_concepts = []  # Track concepts where errors were made THIS quiz
    
    for i, q in enumerate(quiz["questions"]):
        concept = q["concept"]
        normalized_concept = normalize_concept(concept)  # Normalize concept name
        
        concept_stats.setdefault(normalized_concept, {"total": 0, "correct": 0, "original_name": concept})
        concept_stats[normalized_concept]["total"] += 1
        
        if i < len(answers) and answers[i].upper() == q["answer"].upper():
            concept_stats[normalized_concept]["correct"] += 1
            correct += 1
        else:
            mistakes += 1
            # Track concepts where mistakes were made
            if normalized_concept not in error_concepts:
                error_concepts.append(normalized_concept)

    state["last_quiz_had_mistakes"] = mistakes > 0
    state["error_concepts"] = error_concepts  # Store error concepts for adaptive quiz
    
    # Update student model - improvement-focused approach
    for normalized_concept, stats in concept_stats.items():
        current_mastery = stats["correct"] / stats["total"]
        
        if normalized_concept in state["student_model"]:
            old_mastery = state["student_model"][normalized_concept]
            
            # If current performance is better, move towards it quickly
            if current_mastery > old_mastery:
                # 80% weight to new better performance
                state["student_model"][normalized_concept] = (old_mastery * 0.2) + (current_mastery * 0.8)
            else:
                # If worse, only slightly decrease (give benefit of doubt)
                state["student_model"][normalized_concept] = (old_mastery * 0.7) + (current_mastery * 0.3)
        else:
            state["student_model"][normalized_concept] = current_mastery
    
    # Identify weak concepts (for general tracking)
    state["weak_concepts"] = [c for c, m in state["student_model"].items() if m < 0.6]
    
    # Calculate overall mastery
    if state["student_model"]:
        state["overall_mastery"] = sum(state["student_model"].values()) / len(state["student_model"])
    
    state["attempts"] += 1
    state["quiz_history"].append({
        "quiz": quiz,
        "answers": answers,
        "score": f"{correct}/{len(quiz['questions'])}"
    })
    
    # Get readable concept names for logging
    error_names = [concept_stats[c]["original_name"] for c in error_concepts if c in concept_stats]
    log_action(state, "EVALUATED", f"Score: {correct}/{len(quiz['questions'])} | Mastery: {state['overall_mastery']:.0%} | Errors in: {', '.join(error_names) if error_names else 'None'}")
    state["awaiting_user_input"] = False
    
    return state

# =======================
# CELEBRATION NODE
# =======================
def celebration_node(state: AgentState) -> AgentState:
    """Celebrate student success"""
    print("[CELEBRATION NODE] Success!")
    log_action(state, "SUCCESS", "🎉 Student achieved mastery!")
    
    state["user_message"] = f"""
🎉 CONGRATULATIONS! 🎉

You've achieved mastery in {state['topic']}!

Final Stats:
- Overall Mastery: {state['overall_mastery']:.0%}
- Attempts: {state['attempts']}
- Concepts Mastered: {len(state['student_model'])}

Concept Breakdown:
""" + "\n".join([f"  • {c}: {m:.0%}" for c, m in state["student_model"].items()])
    
    return state

# =======================
# ROUTING FUNCTION
# =======================
def route_next_action(state: AgentState) -> Literal["diagnose", "teach", "adaptive_quiz", "evaluate", "celebrate", "end"]:
    """Route to next node based on agent's decision"""
    
    if state["session_complete"]:
        return "celebrate"
    
    if state["awaiting_user_input"]:
        return "end"  # Stop and wait for user
    
    action = state["next_action"]
    print(f"[ROUTING] Going to: {action}")
    
    if action == "diagnose":
        return "diagnose"
    elif action == "teach":
        return "teach"
    elif action == "adaptive_quiz":
        return "adaptive_quiz"
    elif action == "celebrate":
        return "celebrate"
    else:
        return "end"

# =======================
# BUILD AGENT GRAPH
# =======================
def build_agent():
    """Build the autonomous agent workflow"""
    
    workflow = StateGraph(AgentState)
    
    # Add nodes
    workflow.add_node("decide", agent_decision_node)
    workflow.add_node("diagnose", diagnostic_node)
    workflow.add_node("teach", teaching_node)
    workflow.add_node("adaptive_quiz", adaptive_quiz_node)
    workflow.add_node("evaluate", evaluation_node)
    workflow.add_node("celebrate", celebration_node)
    
    # Set entry point
    workflow.set_entry_point("decide")
    
    # Add conditional edges
    workflow.add_conditional_edges(
        "decide",
        route_next_action,
        {
            "diagnose": "diagnose",
            "teach": "teach",
            "adaptive_quiz": "adaptive_quiz",
            "celebrate": "celebrate",
            "end": END
        }
    )
    
    # After each action, return to decision node
    workflow.add_edge("diagnose", END)  # Wait for user input
    workflow.add_edge("teach", "adaptive_quiz")  # Continue to quiz after teaching
    workflow.add_edge("adaptive_quiz", END)  # Wait for user input
    workflow.add_edge("evaluate", "decide")  # Continue autonomous loop
    workflow.add_edge("celebrate", END)
    
    return workflow.compile()

# =======================
# GRADIO UI
# =======================
agent_graph = build_agent()
current_state = None

def start_session(topic):
    """Initialize new learning session"""
    global current_state
    
    if not topic:
        return "⚠️ Please enter a topic first!"
    
    print(f"\n[SESSION START] Topic: {topic}")
    
    current_state = {
        "topic": topic,
        "student_model": {},
        "learning_goal": f"Achieve 80%+ mastery in {topic}",
        "current_quiz": {},
        "quiz_history": [],
        "student_answers": [],
        "weak_concepts": [],
        "error_concepts": [],
        "attempts": 0,
        "overall_mastery": 0.0,
        "agent_logs": [],
        "next_action": "",
        "session_complete": False,
        "awaiting_user_input": False,
        "user_message": "",
        "last_quiz_had_mistakes": False
    }
    
    try:
        # Run agent autonomously
        result = agent_graph.invoke(current_state)
        current_state = result
        return format_state_display(current_state)
    except Exception as e:
        print(f"[ERROR] {str(e)}")
        return f"❌ Error starting session: {str(e)}\n\nPlease check your API key and internet connection."

def submit_answers(answers):
    """Submit quiz answers and let agent continue autonomously"""
    global current_state
    
    if not current_state:
        return "⚠️ Please start a session first!"
    
    if not answers:
        return "⚠️ Please enter your answers!"
    
    print(f"\n[SUBMIT ANSWERS] {answers}")
    
    # Parse answers
    current_state["student_answers"] = [a.strip().upper() for a in answers.split(",")]
    current_state["awaiting_user_input"] = False
    
    try:
        # First evaluate
        current_state = evaluation_node(current_state)
        
        # Then let agent decide and continue
        result = agent_graph.invoke(current_state)
        current_state = result
        
        return format_state_display(current_state)
    except Exception as e:
        print(f"[ERROR] {str(e)}")
        return f"❌ Error: {str(e)}"

def format_state_display(state):
    """Format state for display"""
    output = f"### 🎯 Learning Goal: {state['learning_goal']}\n\n"
    output += f"**Overall Mastery:** {state['overall_mastery']:.0%} | **Attempts:** {state['attempts']}\n\n"
    
    # Agent logs
    output += "### 🤖 Agent Activity Log:\n"
    for log in state["agent_logs"][-6:]:
        output += f"- **{log['timestamp']}** [{log['action']}]: {log['details']}\n"
    
    output += "\n---\n\n"
    
    # Current quiz or message
    if state.get("user_message"):
        output += state["user_message"] + "\n\n"
    
    if state["awaiting_user_input"] and state.get("current_quiz") and "questions" in state["current_quiz"]:
        output += "### 📝 Current Quiz:\n\n"
        for q in state["current_quiz"]["questions"]:
            output += f"**Q{q['id']}.** {q['question']}\n"
            for i, opt in enumerate(q['options']):
                output += f"  {chr(65+i)}) {opt}  \n"
            output += "\n"
        output += "*Enter your answers as comma-separated letters (e.g., A,B,C)*\n\n"
    
    # Student model - display normalized concepts in a readable format
    if state["student_model"]:
        output += "\n### 🧠 Your Knowledge Map:\n"
        for concept, mastery in state["student_model"].items():
            bar = "█" * int(mastery * 10) + "░" * (10 - int(mastery * 10))
            # Make concept more readable: add spaces before capitals, capitalize first letter
            readable_concept = re.sub(r'([a-z])([A-Z])', r'\1 \2', concept).capitalize()
            output += f"- **{readable_concept}**: {bar} {mastery:.0%}\n"
    
    return output

# Build UI
with gr.Blocks(theme=gr.themes.Soft()) as demo:
    gr.Markdown("# 🤖 AI Tutoring System ")
    gr.Markdown("*Autonomous agent that plans, teaches, and adapts based on your learning*")
    
    with gr.Row():
        topic_input = gr.Textbox(label="Learning Topic", placeholder="e.g., Python Loops", value="Python Loops")
        start_btn = gr.Button("🚀 Start Learning Session", variant="primary")
    
    state_display = gr.Markdown(value="Enter a topic and click Start to begin!")
    
    with gr.Row():
        answer_input = gr.Textbox(label="Your Answers (comma-separated)", placeholder="A,B,C,D")
        submit_btn = gr.Button("✅ Submit Answers", variant="secondary")
    
    start_btn.click(start_session, inputs=topic_input, outputs=state_display)
    submit_btn.click(submit_answers, inputs=answer_input, outputs=state_display)

demo.launch(server_name="0.0.0.0", server_port=int(os.getenv("PORT", 7860)))