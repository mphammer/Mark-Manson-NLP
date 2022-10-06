import matplotlib.pyplot as plt
from wordcloud import WordCloud

def create_word_cloud(long_string):
    #  Create a WordCloud object
    wordcloud = WordCloud(background_color="white", width=1600, height=800, max_words=100, contour_width=3, contour_color='steelblue')
    # Generate a word cloud
    wordcloud.generate(long_string)
    # Visualize the word cloud
    plt.figure(figsize = (20,10), facecolor = None)
    plt.imshow(wordcloud)
    plt.axis("off")
    plt.tight_layout(pad = 0)
    plt.show()