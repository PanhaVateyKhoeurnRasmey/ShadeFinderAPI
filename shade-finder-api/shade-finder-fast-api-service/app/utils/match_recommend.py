import pandas as pd

# Load the product recommendation data
df = pd.read_csv('recommendation.csv')

def get_recommendation(undertone: str, tone_label: str):
    """
    Find the best match for the given undertone and tone_label in the dataset.

    Args:
    - undertone (str): The predicted undertone ('warm', 'neutral', 'cool')
    - tone_label (str): The predicted tone label ('deep', 'medium-deep', 'medium', 'light-medium', 'light', 'fair')

    Returns:
    - dict: The matched product details as a dictionary.
    """
    tone_label_dict = {"cool": 0, "neutral": 1, "warm": 2}
    matches = df[(df['warmth'] == tone_label_dict[undertone]) & (df['skin_darkness'] == tone_label)]

    if matches.empty:
        return None
    
    # print(matches,"\n")
    # print(matches["recommendedSerum"])
    return matches.to_dict(orient='records')

if __name__ == "__main__":
    # Test case 1: Match found
    result = get_recommendation('warm', 'medium-deep')
    assert result is not None, "Expected a match but got None"
    assert isinstance(result, list), "Expected a list of matches"
    assert len(result) > 0, "Expected at least one match"
    print("All tests passed!")