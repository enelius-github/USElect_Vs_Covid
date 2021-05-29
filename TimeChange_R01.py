

import pandas
import numpy as np
import matplotlib.pyplot as plt

Elec = pandas.read_csv('2016_US_County_Level_Presidential_Results.csv')  # Found at: https://github.com/tonmcg/US_County_Level_Election_Results_08-16
Rona = pandas.read_csv('time_series_covid19_confirmed_US.csv') # Found at: https://github.com/CSSEGISandData/COVID-19/tree/master/csse_covid_19_data/csse_covid_19_time_series
Popu = pandas.read_csv('PopulationEst.csv')  # Found at: https://www.census.gov/data/datasets/time-series/demo/popest/2010s-counties-total.html#par_textimage_70769902
Deat = pandas.read_csv('time_series_covid19_deaths_US.csv')




# Sort by FIPS number: https://en.wikipedia.org/wiki/Federal_Information_Processing_Standards
Elec = Elec.sort_values('combined_fips')
Rona = Rona.sort_values('FIPS')
Deat = Deat.sort_values('FIPS')

GoodStates = Rona['FIPS'].isin(Elec['combined_fips'])
BadStates = GoodStates == False
BadIdx = BadStates[BadStates].index

RonaRed = Rona.drop(BadIdx) # Get Rid of the outliers (Guam, American Samoa, Etc.)
DeatRed = Deat.drop(BadIdx) # Get Rid of outliers (assume rona and deat have same)

GoodStates = Elec['combined_fips'].isin(Rona['FIPS'])
BadStates = GoodStates == False
BadIdx = BadStates[BadStates].index

ElecRed = Elec.drop(BadIdx)

# Get Vote Margins (How much did GOP win by?)
ElecMargin = ElecRed['per_gop'] - ElecRed['per_dem']
ElecArr =  ElecMargin.to_numpy()


## Get Population By County FIPS code
Popu['COUNTY'] # We need leading zeros
Popu['COUNTY'] = Popu['COUNTY'].apply('{:0>3}'.format)

# Add matching FIPS code to first column
PFip = Popu['STATE'].map(str) + Popu['COUNTY'].map(str) 
PFip = PFip.map(int)
Popu.insert(0, 'FIPS', PFip, True)

# Use only the matching fips codes
GoodStates = Popu['FIPS'].map(int).isin(RonaRed['FIPS'].map(int)) # Both must be integers
BadStates = GoodStates == False
BadIdx = BadStates[BadStates].index

PopuRed = Popu.drop(BadIdx)

# Sort by FIPS (probably not necessary, but don't assume anything)
PopuRed = PopuRed.sort_values('FIPS')

# Get Population by FIPS
PopuArr = PopuRed['POPESTIMATE2019'].to_numpy()



# Get rid of Alaska Data in population and election data
ElecArrNoAK = np.append(ElecArr[:67],ElecArr[95:])
PopuArrNoAK = np.append(PopuArr[:67],PopuArr[95:])





# Calculate Weighted Correlation Coefficient
def wmean(x,weights):
    wm = np.sum(x*weights)/np.sum(weights)
    return wm

def wcovar(x,y,w):
    wmx = wmean(x,w)
    wmy = wmean(y,w)
    return np.sum(w*(x-wmx)*(y-wmy))/np.sum(w)


# Preallocate values

D=21 # Number of days we span over


Ron_UW_beta = np.zeros([len(Rona.columns)-(11+D),2])
Ron_UW_r    = np.zeros([len(Rona.columns)-(11+D),1])

Ron_W_beta = np.zeros_like(Ron_UW_beta)
Ron_W_r    = np.zeros_like(Ron_UW_r)

Det_UW_beta = np.zeros_like(Ron_UW_beta)
Det_UW_r    = np.zeros_like(Ron_UW_r)
Det_W_beta  = np.zeros_like(Det_UW_beta)
Det_W_r     = np.zeros_like(Det_UW_r)







#####**** START LOOP HERE ****#####

for i in range(11,len(Rona.columns)-D):
	j=i-11 # save indexing
	
	# Get Corona change over past 14 days
	RonaChange = RonaRed[Rona.columns[i+D]]-RonaRed[Rona.columns[i]]
	RonaArr =  RonaChange.to_numpy()
	RonaArrNoAK = np.append(RonaArr[:67],RonaArr[95:])  # Get red of corona Alaska data
	
	# Get change in deaths over past 14 days
	DeatChange =  DeatRed[Deat.columns[i+D]] - DeatRed[Deat.columns[i]]
	DeatArr = DeatChange.to_numpy()
	DeatArrNoAK = np.append(DeatArr[:67],DeatArr[95:]) # Drop Alaska
		
	# Get Normalized Case Count Array (per 100k residents)
	RonaNormArrNoAK = 1e5*RonaArrNoAK/PopuArrNoAK
	DeatNormArrNoAK = 1e5*DeatArrNoAK/PopuArrNoAK
	
	
	
	# Replace subzero values with zeros
	RonaNormArrNoAK[RonaNormArrNoAK<0]=0
	DeatNormArrNoAK[DeatNormArrNoAK<0]=0
	
	
	# Calculate Unweighted Trendline (y=ax+b) See: https://en.wikipedia.org/wiki/Weighted_least_squares
	X=np.ones([np.size(ElecArrNoAK),2])
	X[:,0]=ElecArrNoAK
	Xt=np.transpose(X)
	y=np.ones([np.size(ElecArrNoAK),1])
	y[:,0]=RonaNormArrNoAK
	# (Xt*X)*beta = Xt*y  -> beta =  (Xt*X)^-1 * Xt*y
	LHS=np.matmul(Xt,X)
	RHS=np.matmul(Xt,y)
	Ron_UW_beta[j,:] = np.matmul(np.linalg.inv(LHS),RHS).flatten()
	
	# Repeat for death data
	y_det = np.ones([np.size(ElecArrNoAK),1])
	y_det[:,0] = DeatNormArrNoAK
	RHS = np.matmul(Xt, y_det)
	Det_UW_beta[j,:] = np.matmul(np.linalg.inv(LHS),RHS).flatten()
	
	# Calculate Unweighted Correlation coefficient See: https://en.wikipedia.org/wiki/Pearson_correlation_coefficient
	Mean_Elec=np.mean(ElecArrNoAK)
	Mean_Rona=np.mean(RonaNormArrNoAK)
	Mean_Deat=np.mean(DeatNormArrNoAK)   # Calculate mean normalized deaths over past 14 days
	Numer_uw = np.sum((ElecArrNoAK-Mean_Elec)*(RonaNormArrNoAK-Mean_Rona))
	Denom_uw = np.sqrt( np.sum( (ElecArrNoAK-Mean_Elec)**2 ) )*np.sqrt( np.sum( (RonaNormArrNoAK-Mean_Rona)**2 ) )
	Ron_UW_r[j,0] = Numer_uw/Denom_uw
	print('UW Rona Correlation coefficient {:} is'.format(j), Ron_UW_r[j,0])
	Numer_uw = np.sum((ElecArrNoAK-Mean_Elec)*(DeatNormArrNoAK-Mean_Deat))
	Denom_uw = np.sqrt( np.sum( (ElecArrNoAK-Mean_Elec)**2 ) )*np.sqrt( np.sum( (DeatNormArrNoAK-Mean_Deat)**2 ) )
	Det_UW_r[j,0] = Numer_uw/Denom_uw
	print('UW Death Correlation coefficient {:} is'.format(j), Det_UW_r[j,0])
	
	#Calculate Trendline Weighted by Population
	W=np.diag(PopuArrNoAK)
	LHS_w = np.matmul( np.matmul(Xt,W), X)
	Ron_RHS_w = np.matmul( np.matmul(Xt,W), y)
	Det_RHS_w = np.matmul( np.matmul(Xt,W), y_det)
	Ron_W_beta[j,:] = np.matmul(np.linalg.inv(LHS_w),Ron_RHS_w).flatten()
	Det_W_beta[j,:] = np.matmul(np.linalg.inv(LHS_w),Det_RHS_w).flatten()
	
	
	Ron_W_r[j,0] = wcovar(ElecArrNoAK,RonaNormArrNoAK,PopuArrNoAK)/np.sqrt( wcovar(ElecArrNoAK,ElecArrNoAK,PopuArrNoAK)*wcovar(RonaNormArrNoAK,RonaNormArrNoAK,PopuArrNoAK))
	print('W Rona Correlation coefficient {:} is'.format(j), Ron_W_r[j,0])
	Det_W_r[j,0] = wcovar(ElecArrNoAK,DeatNormArrNoAK,PopuArrNoAK)/np.sqrt( wcovar(ElecArrNoAK,ElecArrNoAK,PopuArrNoAK)*wcovar(DeatNormArrNoAK,DeatNormArrNoAK,PopuArrNoAK))
	print('W Death Correlation coefficient {:} in'.format(j), Det_W_r[j,0])


# Plot the correlation over time
plt.plot(Ron_W_r,linewidth=2.8)
plt.plot(Ron_UW_r,linewidth=2.8)
plt.plot(Det_W_r,linewidth=2.8)
plt.plot(Det_UW_r,linewidth=2.8)
plt.legend(['Weighted Rona','Unweighted Rona','Weighted Death', 'Unweighted Death'])
plt.xlabel('Days since ' + Rona.columns[11] + ' (Starting Date)')
plt.ylabel('Pearson Correlation Coefficient')
plt.title('Correlation Coefficient over time using 14-Day window')
plt.show()




def PlotRonaDay(i,D=14):
	j=i-11 # save indexing
	
	# Get Corona change over past 14 days
	RonaChange = RonaRed[Rona.columns[i+D]]-RonaRed[Rona.columns[i]]
	RonaArr =  RonaChange.to_numpy()
	RonaArrNoAK = np.append(RonaArr[:67],RonaArr[95:])  # Get red of corona Alaska data
	
	# Get Normalized Case Count Array (per 100k residents)
	RonaNormArrNoAK = 1e5*RonaArrNoAK/PopuArrNoAK
	
	# Replace subzero values with zeros
	RonaNormArrNoAK[RonaNormArrNoAK<0]=0
	xvals=np.linspace(np.min(ElecArrNoAK),np.max(ElecArrNoAK),50) # Make 50 evenly-spaced points between minimum and maximum vote margin 
	Trend_uw = xvals*Ron_UW_beta[i,0] + Ron_UW_beta[i,1]
	Trend_w  = xvals*Ron_W_beta[i,0]  + Ron_W_beta[i,1]
	label_uw = 'Unweighted, r={:5f}'.format(Ron_UW_r[i,0])
	label_w  = 'Weighted, r={:5f}'.format(Ron_W_r[i,0])
	
	plt.plot(xvals, Trend_uw, linewidth=3, color=[0,0.4,0])
	plt.plot(xvals, Trend_w, linewidth=3, color=[0,0.0,0])
	plt.legend([label_uw,label_w])
	plt.xlabel('GOP Margin (loss) in 2016')
	plt.ylabel('Number of Cases Per 100k')
	plt.title('New Cases from ' +Rona.columns[i] + ' to ' + Rona.columns[i+D]+ ' vs. 2016 Election, Alaska Excluded')
	plt.scatter(ElecArrNoAK,RonaNormArrNoAK, c=ElecArrNoAK, alpha=0.6, cmap='coolwarm')
	plt.show()




def PlotDeatDay(i, D=14):
	j=i-11 # save indexing
	
	# Get Corona change over past 14 days
	DeatChange = DeatRed[Rona.columns[i+D]]-DeatRed[Rona.columns[i]]
	DeatArr =  DeatChange.to_numpy()
	DeatArrNoAK = np.append(DeatArr[:67],DeatArr[95:])  # Get red of corona Alaska data
	
	# Get Normalized Case Count Array (per 100k residents)
	DeatNormArrNoAK = 1e5*DeatArrNoAK/PopuArrNoAK
	
	# Replace subzero values with zeros
	DeatNormArrNoAK[DeatNormArrNoAK<0]=0
	xvals=np.linspace(np.min(ElecArrNoAK),np.max(ElecArrNoAK),50) # Make 50 evenly-spaced points between minimum and maximum vote margin 
	Trend_uw = xvals*Det_UW_beta[i,0] + Det_UW_beta[i,1]
	Trend_w  = xvals*Det_W_beta[i,0]  + Det_W_beta[i,1]
	label_uw = 'Unweighted, r={:5f}'.format(Det_UW_r[i,0])
	label_w  = 'Weighted, r={:5f}'.format(Det_W_r[i,0])
	
	plt.plot(xvals, Trend_uw, linewidth=3, color=[0,0.4,0])
	plt.plot(xvals, Trend_w, linewidth=3, color=[0,0.0,0])
	plt.legend([label_uw,label_w])
	plt.xlabel('GOP Margin (loss) in 2016 US Pres Election')
	plt.ylabel('Number of Deaths Per 100k, by county')
	plt.title('New COVID Deaths from ' +Deat.columns[i] + ' to ' + Deat.columns[i+D]+ ' vs. 2016 Election, Alaska Excluded')
	plt.scatter(ElecArrNoAK,DeatNormArrNoAK, c=ElecArrNoAK, alpha=0.6, cmap='coolwarm')
	plt.show()



