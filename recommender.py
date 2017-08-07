import pandas as pd
import scipy.sparse as sparse
import numpy as np
from scipy.sparse.linalg import spsolve
import implicit
import random
from sklearn.preprocessing import MinMaxScaler

class Recommender:
	
	def __init__(self):
		
		self.website_url = 'OnlineRetail.xlsx'	
		self.retail_data = pd.read_excel(self.website_url) # This may take a couple minutes

		# Data clean up
		self.cleaned_retail = self.retail_data.loc[pd.isnull(self.retail_data.CustomerID) == False]
		self.cleaned_retail['CustomerID'] = self.cleaned_retail.CustomerID.astype(int)
		self.item_lookup = self.cleaned_retail[['StockCode', 'Description']].drop_duplicates()
		self.item_lookup['StockCode'] = self.item_lookup.StockCode.astype(str)
		self.cleaned_retail = self.cleaned_retail[['StockCode', 'Quantity', 'CustomerID']]
		self.grouped_cleaned = self.cleaned_retail.groupby(['CustomerID', 'StockCode']).sum().reset_index()
		self.grouped_cleaned.Quantity.loc[self.grouped_cleaned.Quantity == 0] = 1
		self.grouped_purchased = self.grouped_cleaned.query('Quantity > 0')

		self.customerBoughtCount= self.grouped_purchased.groupby("CustomerID").size()

		# Filter the data so that only customers with >= 10 counts are used for ALS

		moreThanTenSeries = self.customerBoughtCount[self.customerBoughtCount >= 10]
		self.grouped_purchasedBig = self.grouped_purchased[self.grouped_purchased["CustomerID"].isin(moreThanTenSeries.index)]
		self.purchase_sparse = self.getMatrix(self.grouped_purchasedBig)
		self.product_train, self.product_test, self.product_users_altered = self.make_train(self.purchase_sparse, pct_test= 0.2)
		self.user_vecs, self.item_vecs = self.get_alternate_least_squares()

		# Get unique customer and item for look up
		self.customers = list(np.sort(self.grouped_purchased.CustomerID.unique())) # get unique customers
		self.products = list(self.grouped_purchased.StockCode.unique()) # get unique products
		

	def getListOfReaders(self):
		'''For now, return the top 50 users by amount'''
		# return [100, 200, 300]
		return self.customerBoughtCount.sort_values(ascending=False).index[:50]


	def getMatrix(self, hitData):
	    customers = list(np.sort(hitData.CustomerID.unique())) # get unique customers
	    products = list(hitData.StockCode.unique()) # get unique products
	    quantity = list(hitData.Quantity) # all of our purchases

	    rows = hitData.CustomerID.astype('category', categories = customers).cat.codes
	    # get the associated row indices 
	    cols = hitData.StockCode.astype('category', categories = products).cat.codes
	    # get the associated column dices 
	    purchases_sparse = sparse.csr_matrix((quantity, (rows,cols)), shape=(len(customers), len(products)))
	    return purchases_sparse 

	def make_train(self, ratings, pct_test = 0.2):
	    '''
	    This function will take in the original user-item matrix and "mask" a percentage of the original ratings where a
	    user-item interaction has taken place for use as a test set. The test set will contain all of the original ratings, 
	    while the training set replaces the specified percentage of them with a zero in the original ratings matrix. 
	    
	    parameters: 
	    
	    ratings - the original ratings matrix from which you want to generate a train/test set. Test is just a complete
	    copy of the original set. This is in the form of a sparse csr_matrix. 
	    
	    pct_test - The percentage of user-item interactions where an interaction took place that you want to mask in the 
	    training set for later comparison to the test set, which contains all of the original ratings. 
	    
	    returns:
	    
	    training_set - The altered version of the original data with a certain percentage of the user-item pairs 
	    that originally had interaction set back to zero.
	    
	    test_set - A copy of the original ratings matrix, unaltered, so it can be used to see how the rank order 
	    compares with the actual interactions.
	    
	    user_inds - From the randomly selected user-item indices, which user rows were altered in the training data.
	    This will be necessary later when evaluating the performance via AUC.
	    '''
	    test_set = ratings.copy() # Make a copy of the original set to be the test set. 
	    test_set[test_set != 0] = 1 # Store the test set as a binary preference matrix
	    training_set = ratings.copy() # Make a copy of the original data we can alter as our training set. 
	    nonzero_inds = training_set.nonzero() # Find the indices in the ratings data where an interaction exists
	    nonzero_pairs = list(zip(nonzero_inds[0], nonzero_inds[1])) # Zip these pairs together of user,item index into list
	    random.seed(0) # Set the random seed to zero for reproducibility
	    num_samples = int(np.ceil(pct_test*len(nonzero_pairs))) # Round the number of samples needed to the nearest integer
	    samples = random.sample(nonzero_pairs, num_samples) # Sample a random number of user-item pairs without replacement
	    user_inds = [index[0] for index in samples] # Get the user row indices
	    item_inds = [index[1] for index in samples] # Get the item column indices
	    training_set[user_inds, item_inds] = 0 # Assign all of the randomly chosen user-item pairs to zero
	    training_set.eliminate_zeros() # Get rid of zeros in sparse array storage after update to save space
	    return training_set, test_set, list(set(user_inds)) # Output the unique list of user rows that were altered  

	def get_alternate_least_squares(self):
		alpha = 15
		return implicit.alternating_least_squares((self.product_train*alpha).astype('double'),
		                                                           factors = 20,
		                                                           regularization = 0.1,
		                                                           iterations = 50)

	def getRec_Items(self, customer_id):
		customers_arr = np.array(self.customers)
		products_arr = np.array(self.products)

		# return ['ItemA', customer_id]
		final_frame = self.rec_items(customer_id, self.purchase_sparse, self.user_vecs, self.item_vecs, customers_arr, products_arr, self.item_lookup , 10)
		return final_frame

	def rec_items(self, customer_id, mf_train, user_vecs, item_vecs, customer_list, item_list, item_lookup, num_items = 10):
	    '''
	    This function will return the top recommended items to our users 
	    
	    parameters:
	    
	    customer_id - Input the customer's id number that you want to get recommendations for
	    
	    mf_train - The training matrix you used for matrix factorization fitting
	    
	    user_vecs - the user vectors from your fitted matrix factorization
	    
	    item_vecs - the item vectors from your fitted matrix factorization
	    
	    customer_list - an array of the customer's ID numbers that make up the rows of your ratings matrix 
	                    (in order of matrix)
	    
	    item_list - an array of the products that make up the columns of your ratings matrix
	                    (in order of matrix)
	    
	    item_lookup - A simple pandas dataframe of the unique product ID/product descriptions available
	    
	    num_items - The number of items you want to recommend in order of best recommendations. Default is 10. 
	    
	    returns:
	    
	    - The top n recommendations chosen based on the user/item vectors for items never interacted with/purchased
	    '''


	    cust_ind = np.where(customer_list == int(customer_id))[0][0] # Returns the index row of our customer id
	    
	    pref_vec = mf_train[cust_ind,:].toarray() # Get the ratings from the training set ratings matrix
	    pref_vec = pref_vec.reshape(-1) + 1 # Add 1 to everything, so that items not purchased yet become equal to 1
	    pref_vec[pref_vec > 1] = 0 # Make everything already purchased zero
	    rec_vector = user_vecs[cust_ind,:].dot(item_vecs.T) # Get dot product of user vector and all item vectors
	    # Scale this recommendation vector between 0 and 1
	    min_max = MinMaxScaler()
	    rec_vector_scaled = min_max.fit_transform(rec_vector.reshape(-1,1))[:,0] 
	    recommend_vector = pref_vec*rec_vector_scaled 
	    # Items already purchased have their recommendation multiplied by zero
	    product_idx = np.argsort(recommend_vector)[::-1][:num_items] # Sort the indices of the items into order 
	    # of best recommendations
	    rec_list = [] # start empty list to store items
	    for index in product_idx:
	        code = item_list[index]
	        rec_list.append([code, item_lookup.Description.loc[item_lookup.StockCode == code].iloc[0]]) 
	        # Append our descriptions to the list
	    codes = [item[0] for item in rec_list]
	    descriptions = [item[1] for item in rec_list]
	    final_frame = pd.DataFrame({'StockCode': codes, 'Description': descriptions}) # Create a dataframe 
	    final_frame = final_frame[['StockCode', 'Description']].to_html()
	    return final_frame.strip('u\n') # Switch order of columns around
		
