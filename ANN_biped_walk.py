#David Klee 8-11-17


#import all necessary libraries
import sys, math, random, time
#import matplotlib.pyplot as plt
import numpy as np
import scipy
import scipy.integrate
import matplotlib.pyplot as plt

#have to go through and add more parameters to mutation breeding
#number of creatures isnt stable
#refine the fitness function to prevent plateauing
#change the method of adding a new connection


###########################################################################################################################
###############################################  classes  #################################################################
class Speciation:
	def __init__(self,c1=1.0,c2=1.0,c3=0.4,comp_threshold=3.0):
		self.c1 = c1
		self.c2 = c2
		self.c3 = c3 # increase to allow finer distinction based on weights
		self.comp_threshold = comp_threshold #threshold of compatibility difference necessary to create new species

class Creature:
	def __init__(self,node_genes=[],connection_genes=[],fitness=0,speciesID=0,gene_matrix= np.array([])):
		self.node_genes = node_genes
		# array of [ ID, descriptor, value, state]
			# ID of the node
			# descriptor: 1 for input, 2 for output, 3 for hidden, 4 for bias
			# value is a holder where calculations are made during simulation
			# state is 1 for enabled, 0 for disabled
		self.connection_genes = connection_genes
		# array of [ ID, node_from, node_to, weight, state, innov. number]
			# ID tells order in which it was added, starting at 0
			# node_from is node ID where it starts
			# node_to is node ID where it ends
			# weight is the multiplier
			# state is 1 for enabled, 0 for disabled
			# innov. number starting at 1, refer to innovation_record
		self.fitness = fitness
		# fitness is a value that is always positive
		self.speciesID = speciesID
		# species ID is number 1 to +inf
		self.gene_matrix = gene_matrix

class Innovation:
	def __init__(self,number,node_from,node_to):
		self.number = number # starts at 1
		self.node_from = node_from # node ID
		self.node_to = node_to #node ID

class Species_Record:
	def __init__(self,speciesID,num_creatures,generation_record):
		self.speciesID = speciesID # starts at 1, goes to +inf
		self.num_creatures = num_creatures
		self.generation_record = generation_record
		# array of [num_gen, num_creatures, max_fitness, mean_fitness]

class Evolution_Record:
	def __init__(self,num_gen,max_fitness,mean_fitness,num_species,num_creatures,fitness_by_species,avg_num_nodes,avg_num_connections,max_speciesID):
		self.num_gen = num_gen #starting at 1
		self.max_fitness = max_fitness
		self.mean_fitness = mean_fitness
		self.num_species = num_species
		self.num_creatures = num_creatures
		self.fitness_by_species = fitness_by_species
		# list of [speciesID,num_creatures,avgFitness,maxFitness]
		self.avg_num_nodes = avg_num_nodes
		self.avg_num_connections = avg_num_connections
		self.max_speciesID = max_speciesID #highest speciesID ever seen

class Breeding_Specs:
	def __init__(self,elitism_threshold=5,kill_off_threshold=8,kill_off_percentage=0.2,mating_selection_pressure=0,crossover_percentage=0.8,crossover_interspecies_percentage=0,crossover_prob_multipoint=0.6,mutation_prob_add_node=0.03,mutation_prob_add_connection=0.05,mutation_prob_mutate_weight=0.9,mutation_weight_range=5,mutation_weight_cap=8):
		self.elitism_threshold = elitism_threshold # size of population needed to keep best
		self.kill_off_threshold = kill_off_threshold 
		self.kill_off_percentage = kill_off_percentage 
		self.mating_selection_pressure = mating_selection_pressure # value between 0 and 1, 1 meaning high pressure to select most fit,0 meaning no pressure
		self.crossover_percentage = crossover_percentage
		self.crossover_interspecies_percentage = crossover_interspecies_percentage
	 	self.crossover_prob_multipoint = crossover_prob_multipoint
	 	self.mutation_prob_add_node = mutation_prob_add_node
	 	self.mutation_prob_add_connection = mutation_prob_add_connection
	 	self.mutation_prob_mutate_weight = mutation_prob_mutate_weight
	 	self.mutation_weight_cap = mutation_weight_cap # weights will be restricted from -mutation.weight_cap to mutation.weight_cap
	 	self.mutation_weight_range = mutation_weight_range # random distribution with width mutation.weight_range, centered on 0. mutation range of 5 will give random distribution from -2.5 to 2.5

###########################################################################################################################
###########################################################################################################################
###########################################################################################################################


###########################################################################################################################
##############################################  functions  ################################################################
def initializePopulation(pop_size,num_inputs,num_outputs,speciation):
# creates a population of 'pop_size' Creatures
# gives randomized genome to each creature in population
# sorts the creatures by species according to genome (node structure and weights)
# assigns species ID
# updates species record
	innovation_record = []
	population = [Creature() for i in range(pop_size)] # create the population of creatures
	for x in population: # initialize the genome
		x.node_genes = []
		x.connection_genes = []
		# add the nodes: all inputs, outputs + 1 bias
		for input_id in range(num_inputs): # add input nodes
			x.node_genes.append([input_id, 1,0,1])		
		x.node_genes.append([num_inputs,4,1,1]) #add bias node, value starts as 1
		for output_id in range(num_outputs): # add output nodes
			x.node_genes.append([output_id+num_inputs+1,2,0,1])
		# add connections: number of inputs
		for cnxn in range(num_inputs):
			node_from_ = random.choice([a[0] for a in x.node_genes if (a[1]==1 or a[1]==4)])
			node_to_ = random.choice([a[0] for a in x.node_genes if a[1]==2])
			weight = random.uniform(-2.5,2.5)
			same_innov = [a for a in innovation_record if a.node_from == node_from_ and a.node_to == node_to_]
			if same_innov == []: # if unique
				new_innov = Innovation(len(innovation_record)+1,node_from_,node_to_)
				innovation_record.append(new_innov)
				x.connection_genes.append([len(x.connection_genes), node_from_,node_to_,weight,1,len(innovation_record)])
			else: #repeat
				match = [gene for gene in x.connection_genes if gene[5] == same_innov[0].number]
				if len(match) == 0: #not a unique innovation in populaiton, but is unique in this creatures genome
					x.connection_genes.append([len(x.connection_genes),node_from_,node_to_,weight,1,same_innov[0].number])
				else: #not a unique innovation in populaiton, and also a repeat in this creature
					x.connection_genes[match[0][0]][3] += weight #update the weight to the repeated gene

	population, species_record = sortSpeciesFirst(population,speciation)
	return population, species_record, innovation_record

def sortSpeciesFirst(population,speciation):
	# assign the first creature speciesID = 1
	# go through all the remaining creatures, if they match within tolerance assign

	population[0].speciesID = 1
	species_record = []
	species_record.append(Species_Record(1,1,[]))
	model_creatures_ID = [0]
	for creature in population[1:]: # first creature has already been assigned
		creatureID = population.index(creature)
		species_assigned_flag = False
		new_species_flag = False
		species_tested = 0
		while species_assigned_flag == False:
			species_tested += 1
			if species_tested > len(species_record):
				new_species_flag = True
			else:
				model_creature = population[model_creatures_ID[species_tested-1]]
				comp_dist = getCompatibilityDifference(model_creature,creature,speciation)
				if comp_dist > speciation.comp_threshold:
						species_record[species_tested-1].num_creatures += 1
						creature.speciesID = species_tested
						species_assigned_flag = True

			if new_species_flag == True:
				species_record.append(Species_Record(species_tested,1,[]))
				creature.speciesID = species_tested
				model_creatures_ID.append(creatureID)
				species_assigned_flag = True
		# end of while loop
	#end of foor loop
	return population, species_record
# end of function

def getCompatibilityDifference(model_creature,test_creature,speciation):
	c1 = speciation.c1
	c2 = speciation.c2
	c3 = speciation.c3
	model_genes = model_creature.connection_genes
	test_genes = test_creature.connection_genes	
	max1 = max([a[5] for a in model_genes])
	max2 = max([a[5] for a in test_genes])
	min_innov_number = min(max1,max2)
	max_innov_number = max(max1,max2)
	model_genes = makeGenesAligned(model_creature.connection_genes,max_innov_number)
	test_genes = makeGenesAligned(test_creature.connection_genes,max_innov_number)
	num_overlap = 0
	W_val = 0.0
	W_num = 0
	E = 0 # number of excess genes
	D = 0 # number of disjoint genes
	N = 0 # total number of E, D, W_num
	for a in range(max_innov_number):  # [ ID, node_from, node_to, weight, state, innov. number]
		if model_genes[a] != [] or test_genes[a] != []: # we dont want to deal with both being missing
			N += 1
			if model_genes[a] == [] or test_genes[a] == []: #disjoint
				D += 1
			elif a >= min_innov_number: #excess
				E += 1
			else: # overlap, regardless of them being disabled or not
				W_num += 1
				W_val += abs(model_genes[a][3]-test_genes[a][3])

	if W_num == 0: 
		W_num = 0.001 #to avoid dividing by zero
	W = W_val/W_num# average weight difference of matching genes
	comp_dist = c1*E/N + c2*D/N + c3*W
	return comp_dist

def makeGenesAligned(cnxn_genes,max_innov_number):
	aligned_genes = []
	for a in range(max_innov_number):
		a = a+1
		l = [x for x in cnxn_genes if x[5] == a]
		if l == []:
			aligned_genes.append(l)
		else:
			aligned_genes.append(l[0])
	return aligned_genes


def sortSpecies(new_population,old_population,species_record,speciation,evolution_record):
	# reset species_record.num_creatures
	# choose representative creatures for each species in old_population
	# cycle through each creature in new_population
		# compare to all representative creatures 
		# if match then change creature.speciesID and add 1 to species_record.num_creatures
		# if not match then create new species and change creature.speciesID and species_record
	for record in species_record:
		record.num_creatures = 0
	creatures_by_species = [[a for a in old_population if a.speciesID == b] for b in range(1,len(species_record)+1)]
	representative_creatures = [chooseRandomItem(a) for a in creatures_by_species] # index 0 is speciesID 1
	# for speciesID X, if there are no more creatures in old_population then the item in the list is []

	for creature in new_population: #cycle through every creature
		creatureID = new_population.index(creature)
		con_genes_creature = creature.connection_genes
		species_assigned_flag = False
		new_species_flag = False
		species_tested = 0
		new_species_made = 0
		while species_assigned_flag == False: # cycle through every species to compare
			species_tested += 1
			if species_tested > len(species_record):
				new_species_flag = True
			elif representative_creatures[species_tested-1] != []: # this condition skips extinct species
				model_creature = representative_creatures[species_tested-1]
				comp_dist = getCompatibilityDifference(model_creature,creature,speciation)
				if comp_dist > speciation.comp_threshold:
						species_record[species_tested-1].num_creatures += 1
						creature.speciesID = species_tested
						species_assigned_flag = True

			if new_species_flag == True:
				creature.speciesID = evolution_record[-1].max_speciesID+1+new_species_made
				species_record.append(Species_Record(creature.speciesID,1,[]))
				new_species_made += 1
				representative_creatures.append(creature)
				species_assigned_flag = True
		# end of while loop
	#end of for loop
	return new_population, species_record
# end of function

def chooseRandomItem(lst):
	# this allows random choice to be done on a list that may be empty
	if len(lst) > 0:
		return random.choice(lst)
	else:
		return []


def getFitness(creature):
	# simulate the dynamics
	# calculate fitness based on performance
	# assign fitness
	t,Y = simulateDynamics(creature)
	fitness = calculateFitness(t,Y)
	creature.fitness = fitness
	return creature

def calculateFitness(t,Y):
	time_alive = t[-1]
	array_angle = [math.fabs(x[0]) for x in Y]
	sum_angle = sum(array_angle)
	fitness = time_alive/sum_angle+time_alive/2
	return fitness

def simulateDynamics(creature):
	creature = formGeneMatrix(creature)
	# set constants
	y0 = [0.05,0.2]
	t0 = 0
	t1 = 2
	dt = 0.01
	r = scipy.integrate.ode(getDynamics).set_integrator('dopri5')
	r.set_initial_value(y0, t0).set_f_params(creature)
	event_flag = True
	t = []
	Y = []
	while r.successful() and r.t < t1 and event_flag:
		t.append(r.t)
		Y.append(r.y) #list of arrays
		r.integrate(r.t+dt)
		event_flag = limitODE(r.y)
	return t,Y

def limitODE(Y):
	#this does the work of an event function in matlab
	if np.absolute(Y[0]) > math.pi/2 or np.absolute(Y[1]) > 10:
		within_limits = False
	else:
		within_limits = True
	return within_limits #true to continue the simulation

def getDynamics(t,Y,creature):
	# Y = [th, dth]
	# dY = [dth, ddth]
	m,g,l,I = getPhysicalParameters()
	b = 0.1
	U = feedforward(Y,creature)
	dY = [Y[1], (-m*g*l*np.sin(Y[0])-b*Y[1]+U)/I]
	dY = np.asarray(dY)
	return dY

def formGeneMatrix(creature): 
	'''gene matrix:
				 H1   ...  Hx   O1  ...   O9
			I1  [ w  , w  , w  , w  , w ,  w ;
			...   w  , w  , w  , w  , w ,  w ;
			B1    w  , w  , w  , w  , w ,  w ;
			H1    w  , w  , w  , w  , w ,  w ;
			...   w  , w  , w  , w  , w ,  w ;
			H8    w  , w  , w  , w  , w ,  w ]
	'''
	node_genes = creature.node_genes # [ ID, descriptor, value, state] :: 1 for input, 2 for output, 3 for hidden, 4 for bias
	cnxn_genes = creature.connection_genes # [ ID, node_from, node_to, weight, state]
	matrix_input_rows = [a[0] for a in node_genes if a[1] == 1 or a[1] == 3 or a[1] == 4] #provides ID
	matrix_output_columns = [a[0] for a in node_genes if a[1] == 2 or a[1] == 3]
	len_rows = len(matrix_input_rows)
	len_columns = len(matrix_output_columns)
	num_input = len([a for a in node_genes if a[1] == 1])
	num_output = len([a for a in node_genes if a[1] == 2])
	num_hidden = len([a for a in node_genes if a[1] == 3])
	num_bias = len([a for a in node_genes if a[1] == 4])
	new_gene_matrix = np.zeros((len_rows,len_columns)) #initialize array structure
	for i in range(len_rows): #go through all input ID's 
		input_ID = matrix_input_rows[i]
		# find all cnxn genes with this as a node from, and place node 
		end_of_cnxn_nodes = [a for a in cnxn_genes if a[1] == input_ID]
		for end_node in end_of_cnxn_nodes:
			new_gene_matrix[i][end_node[2]-num_input-num_bias] = end_node[3]
	creature.gene_matrix = new_gene_matrix
	return creature		

def feedforward(vector_in,creature):
	num_input_bias = len([a for a in creature.node_genes if a[1] == 1 or a[1] == 4])
	num_output = len([a for a in creature.node_genes if a[1] == 2])
	values_bias_hidden = [a[2] for a in creature.node_genes if a[1] == 3 or a[1] == 4]
	vector_in = np.append(vector_in,values_bias_hidden)
	vector_in = np.array(vector_in) # row vector
	gene_matrix = creature.gene_matrix
	vector_out = np.dot(vector_in,gene_matrix)
	for a in range(len(vector_out)):
		creature.node_genes[a+num_input_bias][2] = transferFunction(vector_out[a]) #change value
	output = vector_out[-num_output:]
	return output

def transferFunction(value):
	output = 1/(1+math.exp(-4.9*value))
	return output

def getPhysicalParameters():
	m = 1
	g = 9.8
	l = 1
	I = m*l^2
	return m,g,l,I

def breedNewPopulation(old_population,species_record,speciation,breeding_specs,evolution_record,innovation_record):
	pop_size = len(old_population)
	new_population = []
	# decide how many offspring each species can make
	# cycle by species and produce given number of offspring
	# assign species to new offspring (sortSpecies)
	alive_species_record = [a for a in species_record if a.num_creatures > 0]
	num_active_species = len(alive_species_record)
	total_fitness = sum([a.generation_record[-1][3] for a in alive_species_record]) #sum of mean fitness
	percent_fitness_by_species = [[alive_species_record[a].speciesID,alive_species_record[a].num_creatures,alive_species_record[a].generation_record[-1][3]/total_fitness] for a in range(num_active_species)]
	# list of [speciesID,num_creatures,percent of total fitness]
	offspring_by_species = []
	overflow = 0
	for item in percent_fitness_by_species: #this can be condensed down at a later point
		speciesID,num_creatures,percent_total_fitness = item
		offspring_by_species.append(math.floor(percent_total_fitness*pop_size))
		overflow += percent_total_fitness*pop_size % 1
		if overflow > 1:
			overflow-=1
			offspring_by_species[-1]+=1
	if sum(offspring_by_species) < pop_size:
		offspring_by_species[-1] += pop_size-sum(offspring_by_species)

	for item in percent_fitness_by_species: # for every alive species
		speciesID,num_creatures,percent_total_fitness = item
		creatures_by_species = [a for a in old_population if a.speciesID == speciesID]
		creatures_by_species_fitness = [a.fitness for a in creatures_by_species]
		offspring_allowed = offspring_by_species[percent_fitness_by_species.index(item)]
		if offspring_allowed > 0:
			mating_creatures_ranked = sorted(creatures_by_species, key=lambda creature: creature.fitness,reverse=True) #descending order
			
			# elitism, to keep best performing creatures in large enough species
			if num_creatures >= breeding_specs.elitism_threshold:
				index_max_fitness = creatures_by_species_fitness.index(max(creatures_by_species_fitness))
				crtr = creatures_by_species[index_max_fitness] #creature with max fitness in species
				new_population.append(Creature(crtr.node_genes,crtr.connection_genes,0,0))
				offspring_allowed -= 1
			
			# kill_off, remove lowest performing individuals from passing on genome if large enough species
			if num_creatures >= breeding_specs.kill_off_threshold and math.ceil(breeding_specs.kill_off_percentage*num_creatures) >= 1:
				num_to_kill = int(math.ceil(breeding_specs.kill_off_percentage*num_creatures))
				mating_creatures_ranked = mating_creatures_ranked[:-num_to_kill] #remove the lowest fitness creatures from mating pool
			# now sample the mating_creatures_ranked to get the right number of 'parents'
			num_crossover = math.floor(breeding_specs.crossover_percentage*offspring_allowed) 
			num_mutation = offspring_allowed - num_crossover
			num_parents = 2*num_crossover+num_mutation
			random_numbers = [random.random() for a in range(int(num_parents))]
			index_parents = [int(math.floor(len(mating_creatures_ranked)*(a*math.exp(-breeding_specs.mating_selection_pressure*a)))) for a in random_numbers]
			parents = [mating_creatures_ranked[a] for a in index_parents]
			parents = random.sample(parents,len(parents)) # to randomize the order because it was ranked previously
			# now do crossover
			num_bred = 0
			while num_bred < offspring_allowed:
				if num_bred < num_crossover:
					parent1,parent2 = parents[0:2] #take last two parents
					parents = parents[2:] #remove the selected parents from the list
					# inherit all node_genes
					node_genes = parent1.node_genes
					new_creature_nodes,new_creature_genes = crossover(parent1,parent2,breeding_specs)
					new_creature = Creature(node_genes=new_creature_nodes,connection_genes=new_creature_genes)
				else: # carry genes on from parent
					parent1 = parents[-1]
					parents = parents[:-1] #remove selected parent from the list
					new_creature = Creature(node_genes=parent1.node_genes,connection_genes=parent1.connection_genes)
				# now we mutate on the new_creatures genome
				random_number = random.random()
				if random_number < breeding_specs.mutation_prob_add_node: #[ ID, descriptor, value, state]
					new_creature.node_genes.append([len(new_creature.node_genes),3,0,1])
				if random_number < breeding_specs.mutation_prob_add_connection: #[ ID, node_from, node_to, weight, state, innov_number]
					ID = len(new_creature.connection_genes)
					node_from_ = random.choice([a[0] for a in new_creature.node_genes if a[1] == 1 or a[1] == 3 or a[1] == 4]) # comes from input, hidden or bias nodes
					node_to_ = random.choice([a[0] for a in new_creature.node_genes if a[1] == 2 or a[1] == 3]) # goes to output or hidden
					if len([a for a in new_creature.connection_genes if a[1] == node_from_ and a[2] == node_to_]) == 0: #this connection does not already exist in the creature
						#find innovation_number
						weight = 10*random.random()-5
						same_innov = [a for a in innovation_record if a.node_from == node_from_ and a.node_to == node_to_]
						if same_innov == []: #unique, new innovation!
							new_innov = Innovation(len(innovation_record)+1,node_from_,node_to_)
							innovation_record.append(new_innov)
							new_creature.connection_genes.append([ID, node_from_,node_to_,weight,1,len(innovation_record)])
						else: #not new innovation
							new_creature.connection_genes.append([ID,node_from_,node_to_,weight,1,same_innov[0].number])
				for conn in new_creature.connection_genes:
					random_number = random.random()
					if random_number < breeding_specs.mutation_prob_mutate_weight:
						conn[3] += breeding_specs.mutation_weight_range*(random.random()-0.5)
						conn_weight_capped = min(max(-breeding_specs.mutation_weight_cap,conn[3]),breeding_specs.mutation_weight_cap)
						conn[3] = conn_weight_capped
				new_population.append(new_creature)
				num_bred += 1
			# end of while

	new_population, species_record = sortSpecies(new_population,old_population,species_record,speciation,evolution_record)

	return new_population,species_record

def crossover(parent1,parent2,breeding_specs): # returns: nodes, genes
	# disjoint and excess genes in the more fit parent are passed on
	# connection_gene = [ ID, node_from, node_to, weight, state, innov. number]
	if parent1.fitness >= parent2.fitness:
		more_fit = 1
	else:
		more_fit = 2
	max1 = max([a[5] for a in parent1.connection_genes])
	max2 = max([a[5] for a in parent2.connection_genes])
	max_innov_number = max(max1,max2)
	min_innov_number = min(max1,max2)
	genes1 = makeGenesAligned(parent1.connection_genes,max_innov_number)
	genes2 = makeGenesAligned(parent2.connection_genes,max_innov_number)
	genes_new = [] #it is in condensed form, not aligned
	for gene1,gene2 in zip(genes1,genes2):
		if gene1 == [] and gene2 == []: #dont evaluate
			pass
		elif gene2 == [] and more_fit == 1: # it is a disjoint gene in genes1 and parent 1 is more fit
			genes_new.append(gene1)
		elif gene1 == [] and more_fit == 2: # it is a disjoint gene in genes2 and parent 2 is more fit
			genes_new.append(gene2)
		elif gene1 == [] or gene2 == []: #ignore this case
			pass
		elif gene1[5] > min_innov_number and more_fit == 1: #it is an excess gene in genes1 and parent1 is more fit
			genes_new.append(gene1)
		elif gene2[5] > min_innov_number and more_fit == 2: #if is an excess gene in genes2
			genes_new.append(gene2)
		else: # matching gene, randomly choose one to pass on
			if random.random() < 0.5:
				genes_new.append(gene1)
			else:
				genes_new.append(gene2)
	# now make sure the ID of each gene goes in order
	for gene in genes_new:
		gene[0] = genes_new.index(gene)
	# for nodes, just pass the parent's nodes which is longer
	if len(parent1.node_genes) >= len(parent1.node_genes):
		nodes = parent1.node_genes
	else:
		nodes = parent2.node_genes
	return nodes, genes_new


def updateGenerationRecord(population,species_record,num_gen):
	# array of [num_gen, num_creatures, max_fitness, mean_fitness]
	for record in species_record:
		creatures_in_species = [x for x in population if x.speciesID == record.speciesID]
		num_creatures = len(creatures_in_species)
		if num_creatures > 0: #there are creatures in the species
			max_fitness = max([x.fitness for x in creatures_in_species])
			mean_fitness = sum([x.fitness for x in creatures_in_species])/num_creatures
			record.generation_record.append([num_gen,num_creatures,max_fitness,mean_fitness])
	return species_record

def showTime(last_time):
	if last_time == 0:
		str1 = 'Time Elapsed: '
	else:
		str1 = 'Time Elapsed on Last Generation: '
	str2 = str(time.clock()-last_time)
	str3 = ' s'
	print(str1+str2+str3)
	last_time_abs = time.clock()
	return last_time_abs

def updateEvolutionRecord(evolution_record,population,species_record):
	num_gen = len(evolution_record)+1
	max_fitness = max([x.fitness for x in population])
	num_creatures = len(population)
	mean_fitness = sum([x.fitness for x in population])/num_creatures
	num_species = len([x for x in species_record if x.num_creatures > 0])
	avg_num_nodes = sum([len(x.node_genes) for x in population])/num_creatures
	avg_num_connections = sum([len(x.connection_genes) for x in population])/num_creatures
	fitness_by_species = []
	for species in species_record:
		speciesID = species.speciesID
		creatures_in_species = [x for x in population if x.speciesID == speciesID]
		num_creatures_species = len(creatures_in_species)
		if num_creatures_species > 0:
			avg_fitness = sum([x.fitness for x in creatures_in_species])/num_creatures_species
			max_fitness = max([x.fitness for x in creatures_in_species])
			fitness_by_species.append([speciesID,num_creatures_species,avg_fitness,max_fitness])
	if len(evolution_record) == 0:
		max_speciesID = max([a.speciesID for a in population])
	else:
		max_speciesID = max(evolution_record[-1].max_speciesID,max([a.speciesID for a in population]))
	new_record = Evolution_Record(num_gen,max_fitness,mean_fitness,num_species,num_creatures,fitness_by_species,avg_num_nodes,avg_num_connections,max_speciesID)
	evolution_record.append(new_record)
	return evolution_record

def showGenStats(evolution_record):
	str_num_gen = str(evolution_record[-1].num_gen)
	str_num_creatures = str(evolution_record[-1].num_creatures)
	str_num_species = str(evolution_record[-1].num_species)
	str_avg_fitness = str(evolution_record[-1].mean_fitness)
	str_max_fitness = str(evolution_record[-1].max_fitness)
	str_avg_num_nodes = str(evolution_record[-1].avg_num_nodes)
	str_avg_num_connections = str(evolution_record[-1].avg_num_connections)
	print('Generation: ' + str_num_gen)
	print('Population of ' + str_num_creatures + ' creatures among ' + str_num_species + ' species.')
	print('Average Fitness: ' + str_avg_fitness + ' || Max Fitness: ' + str_max_fitness)
	print('Number Nodes: ' + str_avg_num_nodes + ' || Number Connections: ' + str_avg_num_connections)
	print('--------------------------------------------')
	pass

def showBestGenome(population,ax):
	creature = max(population, key= lambda x: x.fitness)
	nodes = creature.node_genes # [ ID, descriptor, value, state]
	connections = creature.connection_genes # [ ID, node_from, node_to, weight, state, innov. number]
	hidden_struct = []

	plt.title('Genome of Most Fit Creature (%d)' % (population.index(creature)+1))

	input_nodes = [node for node in nodes if node[1] == 1]
	output_nodes = [node for node in nodes if node[1] == 2]
	bias_node = [node for node in nodes if node[1] == 4]
	hidden_nodes = [node for node in nodes if node[1] == 3]

	input_nodes_ID = [a[0] for a in input_nodes]
	hidden_nodes_ID = [a[0] for a in hidden_nodes]
	bias_node_ID = bias_node[0][0]
	
	input_nodes_info = [[a[0],0,1-input_nodes.index(a)] for a in input_nodes]
	input_nodes_x = [a[1] for a in input_nodes_info]
	input_nodes_y = [a[2] for a in input_nodes_info]

	bias_node_info = [[a[0],0,input_nodes_y[-1]-1] for a in bias_node]
	bias_node_x = bias_node_info[0][1]
	bias_node_y = bias_node_info[0][2]

	hidden_nodes_info = [[a[0],1,len(hidden_nodes)//2-hidden_nodes.index(a)] for a in hidden_nodes]
	hidden_nodes_x = [a[1] for a in hidden_nodes]
	hidden_nodes_y = [a[2] for a in hidden_nodes]

	output_nodes_info = [[a[0],len(hidden_struct)+1,-output_nodes.index(a)] for a in output_nodes]
	output_nodes_x = [a[1] for a in output_nodes_info]
	output_nodes_y = [a[2] for a in output_nodes_info]

	all_nodes_info = input_nodes_info+bias_node_info+hidden_nodes_info+output_nodes_info

	for cnxn in connections: # [ ID, node_from, node_to, weight, state, innov. number]
		node_from_info = filter(lambda x: x[0]==cnxn[1],all_nodes_info)[0]
		node_to_info = filter(lambda x: x[0]==cnxn[2],all_nodes_info)[0]
		if cnxn[3] >= 0: #positive weight value
			plt.plot([node_from_info[1],node_to_info[1]],[node_from_info[2],node_to_info[2]],'g-',linewidth=cnxn[3])
		else:
			plt.plot([node_from_info[1],node_to_info[1]],[node_from_info[2],node_to_info[2]],'r-',linewidth=-cnxn[3])

	#plot nodes
	plt.plot(input_nodes_x,input_nodes_y,'go')
	plt.plot(bias_node_x,bias_node_y,'gD')
	plt.plot(output_nodes_x,output_nodes_y,'bD')
	plt.plot(hidden_nodes_x,hidden_nodes_y,'ks')

	#plot connections
	ax.xaxis.set_visible(False)
	ax.yaxis.set_visible(False)

def produceGUI(population,evolution_record):
	# graph of species percent over generation
	# graph of avg number of nodes
	# graph of avg fitness & graph of max fitness
	# window with live replays of most fit creatures
	fig = plt.figure(figsize=(12,8))

	# live feed
	ax1 = plt.subplot(221)
	plt.plot([4,5,6])
	plt.plot([3,4,4])
	plt.title('Simulation Goes Here')
	ax1.xaxis.set_visible(False)
	ax1.yaxis.set_visible(False)

	# avg fitness and max fitness
	ax2 = plt.subplot(222)
	avg_fitness, = plt.plot([x.mean_fitness for x in evolution_record],label='Average') #avg fitness
	max_fitness, = plt.plot([x.max_fitness for x in evolution_record],label='Maximum') #max fitness
	legend1 = plt.legend(handles=[avg_fitness, max_fitness],loc=2)
	plt.gca().add_artist(legend1)
	ax2.set_xlabel('Generation')
	ax2.set_ylabel('Fitness')
	ax2.yaxis.set_visible(False)
	plt.xticks(range(len(evolution_record)))
	plt.title('Fitness over Generations')



	#show the nodes of best creature
	ax3 = plt.subplot(223)
	showBestGenome(population,ax3)

	# num_gen,max_fitness,mean_fitness,num_species,num_creatures,fitness_by_species,avg_num_nodes,avg_num_connections
																# fitness_by_species = [speciesID,num_creatures]
	# show species size over time
	ax4 = plt.subplot(224)
	plt.title('Species Size over Generations')
	max_speciesID = evolution_record[-1].max_speciesID
	init_max_speciesID = evolution_record[0].max_speciesID
	y_stack = np.zeros((evolution_record[-1].num_gen,max_speciesID))
	x = np.arange(len(y_stack[:,0]))
	for record in evolution_record:
		fitness_by_species = record.fitness_by_species
		num_gen = record.num_gen
		for species_fitness in fitness_by_species:
			num_creatures = species_fitness[1]
			speciesID = species_fitness[0]
			y_stack[num_gen-1][speciesID-1] = num_creatures

	y_stack = np.cumsum(y_stack,axis=1)
	for a in range(max_speciesID):
		if a == 0:
			ax4.fill_between(x, y_stack[:,a])
		else:
			ax4.fill_between(x, y_stack[:,a], y_stack[:,a-1])
	plt.xticks(range(len(evolution_record)))
	plt.yticks([0,evolution_record[0].num_creatures])


	fig.tight_layout() # controls spacing between subplots
	plt.show()



###########################################################################################################################
###########################################################################################################################
###########################################################################################################################

# set evolution parameters
pop_size = 50

max_num_generations = 3
num_inputs = 2
num_outputs = 1

# create starting population with genomes and species ID assigned
speciation = Speciation()
breeding_specs = Breeding_Specs()
evolution_record = []
population,species_record,innovation_record = initializePopulation(pop_size,num_inputs,num_outputs,speciation)
last_time_abs = showTime(0)

num_gen = 0
while num_gen < max_num_generations:
	num_gen += 1
	for creature in population:
		creature = getFitness(creature)
	
	# update species_record.generation_record 
	species_record = updateGenerationRecord(population,species_record,num_gen)
	evolution_record = updateEvolutionRecord(evolution_record,population,species_record)

	# update GUI
		# graph of species percent over generation
		# graph of avg number of nodes
		# graph of avg fitness 
		# graph of max fitness
		# window with live replays of most fit creatures
	produceGUI(population,evolution_record)
	# produce a new generation from crossover, mutation, elitism
	population,species_record = breedNewPopulation(population,species_record,speciation,breeding_specs,evolution_record,innovation_record)
	
	# save population information, species_record

	# print time spent on last generation
	last_time_abs = showTime(last_time_abs)
	showGenStats(evolution_record)
# end of while

'''   THINGS TO DO
crossover part of breedNewPopulation
add threading to quit while loop
add a thinning of all but 2 species if there is a plateau in average fitness after 10 generations
add activation function
species need to be more continuous
add torque limits
multiprocessing
make a visual representation of the developping neural network: https://www.youtube.com/watch?v=T4EopjWkLtI

for some reason the population size isnt stable
'''

