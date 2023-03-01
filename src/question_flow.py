prompt_flow = {
    "welcome" : "Welcome",
    "excitement" : "Are you excited for the upcoming FIFA World Cup?",
    "fav team?" : "Which team are you supporting in this year's world cup?",
    "fav player" : "Who is your favorite player?",
    "confidence" : "How confident are you about your team's chances?",
    "stylistic analysis?" : "Would you like a stylistic analysis of your replies so far?",
    "other teams" : "Are there any other teams that you like?",
    "other player" : "Who is your favorite player from their team?",
    "other confidence" : "How confident are you about this team's chances?",
}

state_flow = {
    "Welcome to the FIFA World Cup Chatbot! What's your name?" : "Are you excited for the upcoming FIFA World Cup?",
    "Are you excited for the upcoming FIFA World Cup?" : "Which team are you supporting in this year's world cup?",
    "Which team are you supporting in this year's world cup?" : "Who is your favorite player?",
    "Who is your favorite player?" : "How confident are you about your team's chances?",
    "How confident are you about your team's chances?" : "Would you like a stylistic analysis of your replies so far?",
    "Would you like a stylistic analysis of your replies so far?" : ["Would you like to continue?", "Thanks for using, bye!"],
    "Thanks for using, bye!" : "Please refresh the page to chat again.",
    "Are there any other teams that you like?" : "Who is your favorite player from their team?",
    "Who is your favorite player from their team?" : "How confident are you about this team's chances?",
    "How confident are you about this team's chances?" : "Would you like a stylistic analysis of your replies so far?",
    "Would you like to continue?" : ["Are there any other teams that you like?", "Thanks for using, bye!"]
}