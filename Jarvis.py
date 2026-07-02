import anthropic
client = anthropic.Anthropic()
conversation_history=[]
print("Jarivs is online..... Type 'quit' to exit")
while True:
  user_input= input("You: ").strip()
  if user_input.lower()=="quit":
    print("Okay going offile....")
    break
  if not user_input:
    continue

  #adding conversation to the history 
  conversation_history.append({
    "role":"user",
    "content":user_input
  })

  # sending message to the claude

  response= client.messages.create(
    model="claude-sonnet-4-6",
    max_tokens=1024,
    system="You are JARVIS, a helpfull and intelligent AI assistent. Be concise, smart  and sligtly witty - like tony stark's AI.",
    messages=conversation_history
  )

  # extract the reply 
  reply= response.content[0].text

#Adding JARVIS's replies to the history so that it remembers every conversation

conversation_history.append({
  "role":"assistent",
  "content": reply
}) 

print(f"\nJARVIS: {reply}\n")

