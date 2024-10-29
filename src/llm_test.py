from azure.ai.textanalytics import TextAnalyticsClient
from azure.core.credentials import AzureKeyCredential

def authenticate_client():
    ta_credential = AzureKeyCredential("8d3a7f5791c94752be0d314e851efb7b")
    text_analytics_client = TextAnalyticsClient(
            endpoint="https://stelligence.openai.azure.com/", 
            credential=ta_credential)
    return text_analytics_client

client = authenticate_client()

response = client.analyze_sentiment(documents=["Write a tagline for an ice cream shop."])

print(response)