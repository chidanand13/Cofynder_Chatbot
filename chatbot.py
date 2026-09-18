import os
import requests
from dotenv import load_dotenv
from openai import OpenAI

class Chatbot:

    def __init__(self, user_id="u001"):
        # Load environment variables, force override so it reads the new key
        load_dotenv(override=True)

        # Get API key
        api_key = os.getenv("OPENAI_API_KEY")

        if not api_key:
            raise ValueError("OPENAI_API_KEY is missing from .env")

        # Create OpenAI client
        self.client = OpenAI(
            api_key=api_key,
            base_url="https://generativelanguage.googleapis.com/v1beta/openai/"
        )

        # User ID for prototype
        self.user_id = user_id

        # Load context once
        self.profile = self._get_user_profile()
        self.cofynder_info = self._get_cofynder_info()

        # Conversation memory
        self.conversation = []
        self.clear_conversation()

    def _get_user_profile(self):
        """Fetch the user's profile from the dummy backend."""
        try:
            # Assuming dummy backend runs on port 8000
            response = requests.get(f"http://127.0.0.1:8000/profiles/{self.user_id}")
            if response.status_code == 200:
                return response.json()
            return None
        except Exception as e:
            # If backend is not running, just return None
            return None

    def _get_cofynder_info(self):
        """Load cofynder info from file if it exists."""
        try:
            base_dir = os.path.dirname(os.path.abspath(__file__))
            file_path = os.path.join(base_dir, "rag", "cofynder.txt")
            if os.path.exists(file_path):
                with open(file_path, "r", encoding="utf-8") as f:
                    return f.read()
            return None
        except Exception:
            return None

    def send_message(self, user_message):
        # Add the original user message to memory
        self.conversation.append({
            "role": "user",
            "content": user_message
        })

        try:
            # Send complete conversation to OpenAI
            response = self.client.chat.completions.create(
                model="gemini-3.6-flash",
                messages=self.conversation
            )

            # Get AI response
            ai_message = response.choices[0].message.content

            # Save AI response to memory
            self.conversation.append({
                "role": "assistant",
                "content": ai_message
            })

            return ai_message

        except Exception as e:
            # Remove the user message if API request failed
            self.conversation.pop()
            return f"Error: {e}"

    def clear_conversation(self):
        system_prompt = "You are a concise, friendly AI assistant. You will be provided with Context Information containing the User Profile and Cofynder Info. Use this context to personalize your responses (e.g., greet the user by their name from the profile). Keep your answers direct and avoid long theoretical explanations."
        
        context = "Context Information:\n"
        if self.profile:
            context += f"\nUser Profile:\n{self.profile}\n"
        if self.cofynder_info:
            context += f"\nCofynder Info:\n{self.cofynder_info}\n"
            
        system_content = f"{system_prompt}\n\n{context}"
        
        self.conversation = [
            {"role": "system", "content": system_content.strip()}
        ]

    def show_memory(self):
        print("\n--- Conversation Memory ---")

        for message in self.conversation:
            print(f"{message['role']}: {message['content']}")

        print("----------------------------\n")


def main():

    chatbot = Chatbot()

    print("=" * 50)
    print("          AI CHATBOT")
    print("=" * 50)

    print("Commands:")
    print("  new  → Start a new conversation")
    print("  memory → Show current conversation")
    print("  exit → Exit chatbot")
    print()

    while True:

        user_message = input("You: ").strip()

        if not user_message:
            continue

        # Exit
        if user_message.lower() == "exit":
            print("AI: Goodbye!")
            break

        # New conversation
        if user_message.lower() == "new":
            chatbot.clear_conversation()
            print("AI: New conversation started.\n")
            continue

        # Show memory
        if user_message.lower() == "memory":
            chatbot.show_memory()
            continue

        # Send message
        response = chatbot.send_message(user_message)

        print("AI:", response)
        print()


if __name__ == "__main__":
    main()