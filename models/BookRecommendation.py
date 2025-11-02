import pandas as pd
import numpy as np

# import books datasets

books = pd.read_csv("../datasets/Books.csv", low_memory=False)
users = pd.read_csv("../datasets/Users.csv", low_memory=False)
ratings = pd.read_csv("../datasets/Ratings.csv", low_memory=False)

print("book shape: ",books.shape)
print("users shape: ",users.shape)
print("ratings shape: ",ratings.shape)

# print(books.head().to_string())
# print(users.head().to_string())
# print(ratings.head().to_string())

# print(f"Books: (rows, columns) = {books.shape}")
# print(f"Users: (rows, columns) = {users.shape}")
# print(f"Ratings: (rows, columns) = {ratings.shape}")

# print(f"Empty values in books: {books.isnull().sum()}")
# print(f"Empty values in users: {users.isnull().sum()}")
# print(f"Empty values in ratings: {ratings.isnull().sum()}")


# print(f"Duplicated values in books: {books.duplicated().sum()}")
# print(f"Duplicated values in users: {users.duplicated().sum()}")
# print(f"Duplicated values in ratings: {ratings.duplicated().sum()}")

# Books and Ratings merge datasets

ratings_with_name = ratings.merge(books, on="ISBN")
# print(f"Empty value in new rating_with_name datasets: \n{ratings_with_name.isnull().sum()}")
# print(f"Duplicated value in new rating_with_name datasets: {ratings_with_name.duplicated().sum()}")
# print(f"Size of new datasets ratings_with_name: {ratings_with_name.shape}")
# print(ratings_with_name.head().to_string())


num_rating_df = ratings_with_name.groupby('Book-Title').count()['Book-Rating'].reset_index()
num_rating_df.rename(columns={'Book-Rating': "num_ratings"}, inplace=True)

ratings_with_name['Book-Rating'] = pd.to_numeric(ratings_with_name['Book-Rating'], errors='coerce')
avg_rating_df = ratings_with_name.groupby('Book-Title', as_index=False)['Book-Rating'].mean()
avg_rating_df.rename(columns={'Book-Rating': 'avg_ratings'}, inplace=True)

popular_df = num_rating_df.merge(avg_rating_df, on='Book-Title')

popular_df = popular_df[popular_df['num_ratings'] >= 250].sort_values('avg_ratings', ascending=False).head(50)

popular_df = popular_df.merge(books, on='Book-Title').drop_duplicates('Book-Title')[
    ['Book-Title', 'Book-Author', 'Image-URL-M', 'num_ratings', 'avg_ratings']]

# print(popular_df.to_string(),"\n",popular_df.shape)
# titles_array=popular_df['Book-Title'].values
# titles_list = list(titles_array)
# with open('column_names.txt', 'a') as file:
#     file.write(str(titles_list))

# print("Titles written to 'titles.txt' in list format")
# import pickle

# with open("popular.pkl",'wb') as file:
#     pickle.dump(popular_df,file)
#     print("Success")



# Collaborative based recommender
x = ratings_with_name.groupby('User-ID').count()['Book-Rating'] > 200
padhe_likhe_users = x[x].index
#
filtered_rating = ratings_with_name[ratings_with_name['User-ID'].isin(padhe_likhe_users)]

y = filtered_rating.groupby('Book-Title').count()['Book-Rating'] >= 50

famous_books = y[y].index
final_ratings = filtered_rating[filtered_rating['Book-Title'].isin(famous_books)]
pt = final_ratings.pivot_table(index='Book-Title', columns='User-ID', values='Book-Rating')
pt.fillna(0, inplace=True)


from sklearn.metrics.pairwise import cosine_similarity

similarity_score = cosine_similarity(pt)


# print(similarity_score, similarity_score.shape)
# print(pt.head().to_string())

def recommend(books_name):
    index = np.where(pt.index == books_name)[0][0]
    similar_items = sorted(list(enumerate(similarity_score[index])), key=lambda x: x[1], reverse=True)[1:6]
    data = []
    for i in similar_items:
        item = []
        temp_df = books[books['Book-Title'] == pt.index[i[0]]]
        item.extend(list(temp_df.drop_duplicates('Book-Title')['Book-Title'].values))
        item.extend(list(temp_df.drop_duplicates('Book-Title')['Book-Author'].values))
        item.extend(list(temp_df.drop_duplicates('Book-Title')['Image-URL-M'].values))
        data.append(item)
    return data

#
# with open("books.pkl",'wb') as file:
#     pickle.dump(books, file)
#     print("books")
# with open("pt.pkl",'wb') as file:
#     pickle.dump(pt, file)
#     print("pt")
# with open("similarity_score.pkl",'wb') as file:
#     pickle.dump(similarity_score, file)
#     print("similarity")
# print(recommend("Message in a Bottle"))
