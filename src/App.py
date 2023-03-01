from flask import Flask, render_template, request, jsonify
from flask_cors import CORS
import project_p4 as chat
import question_flow

app = Flask(__name__)
CORS(app)

params = chat.initialize()
answers = []
state_flow = question_flow.state_flow
prompt_flow = question_flow.prompt_flow

@app.post("/message")
def message():
    print("Entered message()")
    nxt = ''
    while True:
        
        #get user input and previous prompt
        user_response = request.get_json().get("message")
        if nxt != '':
            prev = nxt
        else:
            prev = request.get_json().get("prompt")
        
        print("\n\nTHE PROMPT IS :", prev[-2],'\n\n')
        print("\n\nTHE TEXT IS :", user_response)
        print("Extracted question:", prev[-2]['message'])
        print("about to call chat.reply()")
        
        curr_prompt = prev[-2]['message']
        if curr_prompt in ["Would you like a stylistic analysis of your replies so far?",
                           "Would you like to continue?"]:
            if "no" in user_response.lower():
                nxt_prompt = state_flow[curr_prompt][1]
            else:
                nxt_prompt = state_flow[curr_prompt][0]
        else:
            nxt_prompt = state_flow[curr_prompt]
        
        response = chat.reply(curr_prompt, user_response, params, answers)
        message = {"answer": response, "nextPrompt": nxt_prompt}
        print("Called chat.reply()")
        print("The response from chat.reply() is:", response)
        print("\n\n\nANSWERS  =  ", answers, "\n\n\n")
        return jsonify(message)

if __name__ == "__main__":
    app.run(debug=True)