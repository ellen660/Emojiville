import openai
import pandas as pd
import torch
import re
from torch.utils.data import Dataset, DataLoader
import tiktoken
import ast

# Set your OpenAI API key here
openai.api_key = "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"

# Function to detect emojis using regex
def contains_emoji(text):
    emoji_pattern = re.compile(
        "["  # Emoji characters
        u"\U0001F600-\U0001F64F"  # emoticons
        u"\U0001F300-\U0001F5FF"  # symbols & pictographs
        u"\U0001F680-\U0001F6FF"  # transport & map symbols
        u"\U0001F700-\U0001F77F"  # alchemical symbols
        u"\U0001F780-\U0001F7FF"  # Geometric Shapes Extended
        u"\U0001F800-\U0001F8FF"  # Supplemental Arrows-C
        u"\U0001F900-\U0001F9FF"  # Supplemental Symbols and Pictographs
        u"\U0001FA00-\U0001FA6F"  # Chess Symbols
        u"\U0001FA70-\U0001FAFF"  # Symbols and Pictographs Extended-A
        u"\U00002702-\U000027B0"  # Dingbats
        u"\U000024C2-\U0001F251"  # Enclosed characters
        "]+", flags=re.UNICODE)
    
    return emoji_pattern.findall(text)

#  Function to get embeddings from OpenAI
def get_chatgpt_embeddings(texts):
    """Fetch embeddings for a list of texts from ChatGPT's API."""
    response = openai.embeddings.create(
            model= "text-embedding-ada-002",
            input=texts
        )
    # breakpoint()
    return response.data[0].embedding # Change this

# Define a custom Dataset class
class SentimentDataset(Dataset):
    def __init__(self, csv_file, debug=False):
        self.data = pd.read_csv(csv_file)
        self.debug = debug
        
    def __len__(self):
        if self.debug == True:
            return 48
        return len(self.data)
    
    def __getitem__(self, idx):
        # Get the text and label for the current index
        sentence = self.data.loc[idx, 'cleaned_text']
        sentiment = self.data.loc[idx, 'sentiment']
        tokenize = self.data.loc[idx, 'tokens']
        words_list = ast.literal_eval(tokenize)
        emojis = [token for token in words_list if contains_emoji(token)]

        # encoding = tiktoken.get_encoding("cl100k_base")  # Use "cl100k_base" for GPT-3.5 and GPT-4

        # # Example sentence
        # sentence = "🧡 AirdropBox event for ecological users is here. A total of 550,000 addresses are eligible for and 5 types of AirDropbox with different scarcity can be issued.\n\n💙Invitation code: 52DC39\n🏆Airdrop Portal:👉"

        # # Tokenize the sentence using tiktoken (similar to Hugging Face tokenizer)
        # token_ids = encoding.encode(sentence)

        # # Convert token ids to tokens (i.e., back to text representations)
        # token_list = encoding.decode(token_ids)

        # # Convert token_ids to a tensor, as you would with Hugging Face's tokenizer
        # token_ids_tensor = torch.tensor(token_ids)

        # if not emojis:
        #     # If no emoji tokens, return an empty tensor and sentiment
        #     raise ValueError("No emoji tokens found in the sentence")
        
        # # Get the embeddings for the entire sentence
        # sentence_embedding = get_chatgpt_embeddings([sentence])

        # Now, extract the embeddings corresponding to emoji tokens
        # First, split the sentence to align emoji tokens with their positions
        # tokenized_sentence = sentence.split()
        
        emoji_embeddings = None

        for emoji in emojis:
            # emoji_token = encoding.encode(emoji)
            # breakpoint()
            if emoji_embeddings is None:
                emoji_embeddings = torch.tensor(get_chatgpt_embeddings([emoji]), dtype=torch.float)
            else:
                emoji_embeddings = emoji_embeddings + torch.tensor(get_chatgpt_embeddings([emoji]), dtype=torch.float)

            # emoji_embeddings.append(get_chatgpt_embeddings([emoji]))
            # if token in tokenized_sentence:
            #     # Find the position of the emoji token in the tokenized sentence
            #     idx = tokenized_sentence.index(token)
            #     # Assuming that the embeddings from OpenAI correspond to the full sentence
            #     # We need to map this token position to the original sentence
            #     emoji_embeddings.append(sentence_embedding[idx])
        
        # Convert sentiment label to tensor
        sentiment_label = torch.tensor(sentiment, dtype=torch.long)
        # breakpoint()
        
        return emoji_embeddings, sentiment_label
    
if __name__ == "__main__":

    # Usage example
    csv_file = 'cleaned_backhand_index_pointing_right_with_sentiment.csv'
    batch_size = 48
    dataset = SentimentDataset(csv_file, debug=True)

    feature, label = dataset.__getitem__(0)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    # # Loop through the dataloader
    for features, labels in dataloader:
        print("Features (Emoji embeddings):", features.shape)
        print("Labels (Sentiment):", labels.shape)
        break
