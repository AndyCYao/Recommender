import pandas as pd
import scipy.sparse as sparse
import numpy as np
from scipy.sparse.linalg import spsolve

class Recommender:
	
	def __init__(self):
		self.website_url = 'OnlineRetail.xlsx'	
		self.retail_data = pd.read_excel(self.website_url) # This may take a couple minutes

		# Data clean up
		self.cleaned_retail = self.retail_data.loc[pd.isnull(self.retail_data.CustomerID) == False]
		self.cleaned_retail['CustomerID'] = self.cleaned_retail.CustomerID.astype(int)
		self.cleaned_retail = self.cleaned_retail[['StockCode', 'Quantity', 'CustomerID']]
		self.grouped_cleaned = self.cleaned_retail.groupby(['CustomerID', 'StockCode']).sum().reset_index()
		self.grouped_cleaned.Quantity.loc[self.grouped_cleaned.Quantity == 0] = 1
		self.grouped_purchased = self.grouped_cleaned.query('Quantity > 0')

		self.customerBoughtCount= self.grouped_purchased.groupby("CustomerID").size()

		# Filter the data so that only customers with >= 10 counts are used for ALS
		moreThanTenSeries = customerBoughtCount[customerBoughtCount >= 10]
		grouped_purchasedSmall = grouped_purchased[grouped_purchased["CustomerID"].isin(lessThanTenSeries.index)] 

	def getListOfReaders(self):
		'''For now, return the top 50 users by amount'''
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