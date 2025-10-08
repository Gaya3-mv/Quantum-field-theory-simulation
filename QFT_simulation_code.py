
# -*- coding: utf-8 -*-
"""
Created on Mon Aug 21 00:47:48 2023

@author: Gayathree
"""

'''self_induced inertias varying with Lambda'''

'''modified program to run as a Qiskit RuntimeJob'''

# using the code from IBM Spring Challenge 2022, Qiskit Youtube videos, IQC 2021 and Qiskit Nature documentation

import qiskit as qkit
import numpy as np
import sys
from qiskit.tools.visualization import plot_histogram
from qiskit.tools.monitor import job_monitor
import math

from qiskit.transpiler import PassManager
from qiskit.transpiler.passes import Unroller, BasisTranslator

np.set_printoptions(threshold=5000)
np.set_printoptions(linewidth=np.inf)

import itertools

from qiskit_nature.operators.second_quantization import QuadraticHamiltonian
from qiskit_nature.operators.second_quantization import FermionicOp

from qiskit_nature.operators.second_quantization import VibrationalOp
from qiskit_nature.mappers.second_quantization import DirectMapper, JordanWignerMapper
from qiskit_nature.converters.second_quantization import QubitConverter

from qiskit.circuit import Parameter

from qiskit.opflow import X, Y, Z, I
from qiskit.opflow import commutator, anti_commutator
from qiskit.opflow import PauliTrotterEvolution, Suzuki
from qiskit.opflow import (StateFn, Zero, One, Plus, Minus, H,
                           DictStateFn, VectorStateFn, CircuitStateFn, OperatorStateFn)
from qiskit.opflow.primitive_ops import PauliSumOp

from qiskit.providers.aer import AerSimulator   

# constants:
bare_coupling = 4.0 # lambda
g = round(bare_coupling/math.sqrt(4*math.pi), 3)
print(f"g:{g}")
m_f = 6.7 
m_b = 1.0
K = 4
n_modals = 3
     
# function for calculating {n|m}:
def nm_fn(b, a):    
    if a == (-b) and a != 0 and b != 0:
        return (1/b)
    else:   
        return 0

b_mapper = DirectMapper()
f_mapper = JordanWignerMapper()

sigma_plus = 0.5*(X + 1j*Y)    # VibrationalOp a (annihilation op)
sigma_minus = 0.5*(X - 1j*Y)    # VibrationalOp a_dag (creation op)
I_plus = 0.5*(I+Z)     # from the paper...
I_minus = 0.5*(I-Z)    # ...2105.12563.pdf   

n_qb = int(math.log(n_modals+1, 2))    # No. of qubits reqr. for the bosonic operator
n = n_qb - 1

cutoff = 2*K + n_qb*K    # this is length of each operator string
                         # (or No. of qubits for a Fock state)

b_creation_op = 0
for p in reversed(range(n+1)):
    q = n-p

    step = 2**(q+1)
    *I_list, = itertools.product([I_plus, I_minus], repeat = p)

    for i,j in zip(range(2**q, n_modals+1, step), I_list):
        if I_list == [()]:
            term_here = (math.sqrt(i)*(sigma_minus)^(sigma_plus^q))
        else:
            I_tensored_term = j[0]
            for k,I_term in enumerate(j[1:]):
                I_tensored_term = ((I_tensored_term) ^ I_term)
                    
            if q == 0:
                term_here = (math.sqrt(i)*(I_tensored_term) ^ (sigma_minus))
            else:
                term_here = (math.sqrt(i)*(I_tensored_term) ^ (sigma_minus)^(sigma_plus^q))
            
        i += step
        b_creation_op += term_here

b_creation_op = b_creation_op.reduce()    # the bosonic creation operator
b_creation_op_Paulis = b_creation_op.primitive.paulis.to_labels(array = True)
b_creation_op_coeffs = np.around(b_creation_op.primitive.coeffs, 4)
b_creation_op_new_list = list(zip(b_creation_op_Paulis, b_creation_op_coeffs))
b_creation_op = PauliSumOp.from_list(b_creation_op_new_list)
print('modified b_creation_op:', b_creation_op)

b_anni_op = b_creation_op.adjoint()    # the bosonic annihilation operator

def op_mapper(index, op, ptype):   
    index -= 1                           
    str_in = 'I'*(2*K + n_qb*K)
    
    if ptype == 'f':
        str_in = str_in[:index] + op + str_in[(index+1):]
        str_in = str_in[::-1]    # reversing the string
        mapped_op = f_mapper.map(FermionicOp([(str_in, 1)]))
    
    elif ptype == 'a':
        index+=K
        str_in = str_in[:index] + op + str_in[(index+1):]
        str_in = str_in[::-1]    # reversing the string
        mapped_op = f_mapper.map(FermionicOp([(str_in, 1)]))
        
    elif ptype == 'b':
        index+=(2*K + index*(n_qb - 1))
        if op == '+':
            b_op = b_creation_op
        elif op == '-':    
            b_op = b_anni_op
        
        mapped_op = (I^(len(str_in[:index])))^(b_op)^(I^(len(str_in[(index+1):])-n_qb+1))
        
    return mapped_op

Lambda = 2048    # cutoff Lambda

def alpha(nn):
    if nn == 1:
        ret_val = -round((sum((nm_fn(nn-m, m-nn) - nm_fn(nn+m, (-m-nn))) for m in range(1, Lambda+1))), 4)
        
    else:
        ret_val = round((sum((nm_fn(nn-m, m-nn) - nm_fn(nn+m, (-m-nn))) for m in range(1, Lambda+1)) 
                        - sum((nm_fn(1-m, m-1) - nm_fn(1+m, (-m-1))) for m in range(1, Lambda+1))), 4)
    return ret_val

def beta(nn):
    return round((sum(((nn/m)*nm_fn(nn-m, m-nn)) for m in range(1, Lambda+1))), 4)
   
def gamma(nn):
    return round((sum(((nn/m)*nm_fn(nn+m, (-m-nn))) for m in range(1, Lambda+1))), 4)
   
H_m = sum((1/n)*(((op_mapper(n, '+', 'b')@op_mapper(n, '-', 'b'))*(m_b**2 + g**2 * alpha(n)))
              + ((op_mapper(n, '+', 'f')@op_mapper(n, '-', 'f'))*(m_f**2 + g**2 * beta(n)))
              + ((op_mapper(n, '+', 'a')@op_mapper(n, '-', 'a'))*(m_f**2 + g**2 * gamma(n))))
          for n in range(1, K+1)) 

H_v = g*m_f*sum((1/math.sqrt(l))*((((op_mapper(k, '+', 'f')@op_mapper(m, '-', 'f')@op_mapper(l, '+', 'b'))+(op_mapper(m, '+', 'f')@op_mapper(k, '-', 'f')@op_mapper(l, '-', 'b')))*(nm_fn(k+l, (-m)) + nm_fn(k, l-m)))
                                + (((op_mapper(k, '+', 'a')@op_mapper(m, '-', 'a')@op_mapper(l, '+', 'b'))+(op_mapper(m, '+', 'a')@op_mapper(k, '-', 'a')@op_mapper(l, '-', 'b')))*(nm_fn(k+l, (-m)) + nm_fn(k, l-m)))
                                - (((op_mapper(k, '-', 'f')@op_mapper(m, '-', 'a')@op_mapper(l, '+', 'b'))+(op_mapper(m, '+', 'a')@op_mapper(k, '+', 'f')@op_mapper(l, '-', 'b')))*(nm_fn(k-l, m) + nm_fn(k, (-l+m))))) 
                for k in range(1, K+1) for l in range(1, K+1) for m in range(1, K+1))

H_s = (g**2)*sum((1/math.sqrt(l))*(1/math.sqrt(n))*(((op_mapper(k, '+', 'f')@op_mapper(m, '-', 'f')@op_mapper(l, '+', 'b')@op_mapper(n, '-', 'b'))*(nm_fn(k-n, l-m) + nm_fn(k+l, (-m-n))))
                                                  + ((op_mapper(k, '+', 'a')@op_mapper(m, '-', 'a')@op_mapper(l, '+', 'b')@op_mapper(n, '-', 'b'))*(nm_fn(k-n, l-m) + nm_fn(k+l, (-m-n))))
                                                 + (((op_mapper(k, '-', 'a')@op_mapper(m, '-', 'f')@op_mapper(l, '+', 'b')@op_mapper(n, '+', 'b'))+(op_mapper(m, '+', 'f')@op_mapper(k, '+', 'a')@op_mapper(n, '-', 'b')@op_mapper(l, '-', 'b')))*nm_fn(l-k, n-m)))
                for k in range(1, K+1) for l in range(1, K+1) for m in range(1, K+1) for n in range(1, K+1))

H_f = (g**2)*sum((1/math.sqrt(l))*(1/math.sqrt(n))*((((op_mapper(k, '+', 'f')@op_mapper(m, '-', 'f')@op_mapper(l, '+', 'b')@op_mapper(n, '+', 'b'))+(op_mapper(m, '+', 'f')@op_mapper(k, '-', 'f')@op_mapper(n, '-', 'b')@op_mapper(l, '-', 'b')))*nm_fn(k+l, n-m))
                                                  + (((op_mapper(k, '+', 'a')@op_mapper(m, '-', 'a')@op_mapper(l, '+', 'b')@op_mapper(n, '+', 'b'))+(op_mapper(m, '+', 'a')@op_mapper(k, '-', 'a')@op_mapper(n, '-', 'b')@op_mapper(l, '-', 'b')))*nm_fn(k+l, n-m))
                                                   + ((op_mapper(k, '+', 'f')@op_mapper(m, '+', 'a')@op_mapper(l, '+', 'b')@op_mapper(n, '-', 'b'))*(nm_fn(k-n, m+l) + nm_fn(k+l, m-n)))
                                                   + ((op_mapper(m, '-', 'a')@op_mapper(k, '-', 'f')@op_mapper(n, '+', 'b')@op_mapper(l, '-', 'b'))*(nm_fn(k-n, m+l) + nm_fn(k+l, m-n))))
                for k in range(1, K+1) for l in range(1, K+1) for m in range(1, K+1) for n in range(1, K+1))

# H_lf = H_m + H_v + H_s + H_f    (will create H_lf after reducing and rounding-off each term)

# Reducing and rounding-off each term:
# (this can also be written as a single code loop over all the terms)

# Reducing and rounding-off H_m:
H_m = H_m.reduce()
H_m_Paulis = H_m.primitive.paulis.to_labels(array = True)
H_m_coeffs = np.around(H_m.primitive.coeffs, 4)
H_m_new_list = list(zip(H_m_Paulis, H_m_coeffs))
H_m = PauliSumOp.from_list(H_m_new_list)

# Reducing and rounding-off H_v:
H_v = H_v.reduce()
H_v_Paulis = H_v.primitive.paulis.to_labels(array = True)
H_v_coeffs = np.around(H_v.primitive.coeffs, 4)
H_v_new_list = list(zip(H_v_Paulis, H_v_coeffs))
H_v = PauliSumOp.from_list(H_v_new_list)

# Reducing and rounding-off H_s:
H_s = H_s.reduce()
H_s_Paulis = H_s.primitive.paulis.to_labels(array = True)
H_s_coeffs = np.around(H_s.primitive.coeffs, 4)
H_s_new_list = list(zip(H_s_Paulis, H_s_coeffs))
H_s = PauliSumOp.from_list(H_s_new_list)

# Reducing and rounding-off H_f:
H_f = H_f.reduce()
H_f_Paulis = H_f.primitive.paulis.to_labels(array = True)
H_f_coeffs = np.around(H_f.primitive.coeffs, 4)
H_f_new_list = list(zip(H_f_Paulis, H_f_coeffs))
H_f = PauliSumOp.from_list(H_f_new_list)

H_lf = H_m + H_v + H_s + H_f
H_lf = H_lf.reduce()

print(f'The light-front Hamiltonian is:{H_lf}')
print(f"H_lf is Hermitian: {H_lf.is_hermitian()}")

print('Evolution:\n')    #code from the Qiskit Opflow tutorial

#init_state = Zero^One^(Zero^8)
#init_state = One^One^(One^8)
##init_state = One^(Zero^4)^One^(Zero^2)    #One fermion and one boson K = 2, n = 3
##init_state = (Zero^One)^2^((Zero)^3)^One    #One fermion, one anti-fermion, and one boson K = 2, n = 3
#init_state = (Zero^One^Zero)^2^(Zero^3)^One^(Zero^2)    #One fermion, one anti-fermion, and one boson K = 3, n = 3

#lambda comparative study:
##init_state = (Zero^11)^One^(Zero^4)    # 1. One boson alone in K = 2 level (max. K = 4, n = 3)
##init_state = Zero^One^(Zero^14)    # 2. One fermion alone in K = 2 level (max. K = 4, n = 3)
##init_state = (Zero^5)^One^(Zero^10)    # 3. One antifermion alone in K = 2 level (max. K = 4, n = 3)
##init_state = Zero^One^(Zero^2)^Zero^One^(Zero^5)^One^(Zero^4)    # 4. One particle of each kind in K = 2 level (f, f_bar, b) (max. K = 4, n = 3)

#lambda comparative study verification (K = 3):
##init_state = (Zero^9)^One^(Zero^2)    # 1. One boson alone in K = 2 level (max. K = 3, n = 3)
##init_state = Zero^One^(Zero^10)    # 2. One fermion alone in K = 2 level (max. K = 3, n = 3)
##init_state = (Zero^4)^One^(Zero^7)    # 3. One antifermion alone in K = 2 level (max. K = 3, n = 3)
##init_state = Zero^One^(Zero^2)^One^(Zero^4)^One^(Zero^2)    # 4. One particle of each kind in K = 2 level (f, f_bar, b) (max. K = 3, n = 3)
##init_state = (Zero^11)^One    # One boson alone in K = 3 level (max. K = 3, n = 3) 

#lambda comparative study (K = 4):
##init_state = (Zero^11)^One^(Zero^4)    # 1. One boson alone in K = 2 level (max. K = 4, n = 3)
init_state = Zero^One^(Zero^14)    # 2. One fermion alone in K = 2 level (max. K = 4, n = 3)
##init_state = (Zero^5)^One^(Zero^10)    # 3. One antifermion alone in K = 2 level (max. K = 4, n = 3)
##init_state = Zero^One^(Zero^2)^Zero^One^(Zero^5)^One^(Zero^4)    # 4. One particle of each kind in K = 2 level (f, f_bar, b) (max. K = 4, n = 3)

##init_state = (Zero^13)^One^(Zero^2)    # One boson alone in K = 3 level (max. K = 4, n = 3)

#----------------------------
#lambda comparative study (K = 4), num_modals = 7:
##init_state = (Zero^13)^One^(Zero^6)    # 1. One boson alone in K = 2 level (max. K = 4, n = 7)
##init_state = Zero^One^(Zero^18)    # 2. One fermion alone in K = 2 level (max. K = 4, n = 7)
##init_state = (Zero^5)^One^(Zero^14)    # 3. One antifermion alone in K = 2 level (max. K = 4, n = 7)
##init_state = Zero^One^(Zero^3)^One^(Zero^7)^One^(Zero^6)    # 4. One particle of each kind in K = 2 level (f, f_bar, b) (max. K = 4, n = 7)
#----------------------------

#lambda comparative study for K = 5:
##init_state = (Zero^13)^One^(Zero^6)    # 1. One boson alone in K = 2 level (max. K = 5, n = 3)
##init_state = Zero^One^(Zero^18)    # 2. One fermion alone in K = 2 level (max. K = 5, n = 3)
##init_state = (Zero^6)^One^(Zero^13)    # 3. One antifermion alone in K = 2 level (max. K = 5, n = 3)
##init_state = Zero^One^(Zero^4)^One^(Zero^6)^One^(Zero^6)    # 4. One particle of each kind in K = 2 level (f, f_bar, b) (max. K = 5, n = 3)

#lambda comparative study for K = 6:
##init_state = (Zero^15)^One^(Zero^8)    # 1. One boson alone in K = 2 level (max. K = 6, n = 3)
##init_state = Zero^One^(Zero^22)    # 2. One fermion alone in K = 2 level (max. K = 6, n = 3)
##init_state = (Zero^7)^One^(Zero^16)    # 3. One antifermion alone in K = 2 level (max. K = 6, n = 3)
##init_state = Zero^One^(Zero^5)^One^(Zero^7)^One^(Zero^8)    # 4. One particle of each kind in K = 2 level (f, f_bar, b) (max. K = 6, n = 3)

#lambda comparative study for K = 7:
##init_state = (Zero^17)^One^(Zero^10)    # 1. One boson alone in K = 2 level (max. K = 7, n = 3)
##init_state = Zero^One^(Zero^26)    # 2. One fermion alone in K = 2 level (max. K = 7, n = 3)
##init_state = (Zero^8)^One^(Zero^19)    # 3. One antifermion alone in K = 2 level (max. K = 7, n = 3)
##init_state = Zero^One^(Zero^6)^One^(Zero^8)^One^(Zero^10)    # 4. One particle of each kind in K = 2 level (f, f_bar, b) (max. K = 7, n = 3)

##init_state = One^One^(Zero^10)    #One proton each in K = 1 and K = 2

print(init_state)

evo_time = Parameter('t')
evolution_op = (evo_time*H_lf).exp_i()

delta_t = 0.2
num_time_slices = 10
print(f'No. of repetitions = {num_time_slices}')
#trotterized_op = PauliTrotterEvolution(trotter_mode='trotter', reps=num_time_slices).convert(evolution_op @ init_state)    # different Trotter mode
trotterized_op = PauliTrotterEvolution(trotter_mode=Suzuki(order=1, reps=num_time_slices)).convert(evolution_op @ init_state)  #order 1
##trotterized_op = PauliTrotterEvolution(trotter_mode=Suzuki(order=2, reps=num_time_slices)).convert(evolution_op @ init_state)  #order 2
trot_op_circ = trotterized_op.to_circuit()
q_reg = qkit.QuantumRegister(16)
c_reg = qkit.ClassicalRegister(16)
trot_op_qc = qkit.QuantumCircuit(q_reg, c_reg)
trot_op_qc.append(trot_op_circ, q_reg)
trot_op_qc = trot_op_qc.bind_parameters({evo_time: delta_t}) # giving parameter inside sampler
trot_op_circ_decomp = trot_op_qc.decompose()
trot_op_circ_decomp.measure(q_reg, c_reg)
#trot_op_circ_decomp.decompose().draw('mpl')
print('Created the Trotter circuit!')

from qiskit_ibm_provider import IBMProvider
from qiskit_ibm_runtime import QiskitRuntimeService, Session, Options, Sampler

service = QiskitRuntimeService(channel='ibm_quantum', instance = 'ibm-q/open/main')

provider = IBMProvider(instance = 'ibm-q/open/main')

shots = 8192

print('Account loaded!')

trot_qc_transpiled = qkit.transpile(trot_op_circ_decomp, optimization_level=3)
print(trot_qc_transpiled.depth())
    
options = Options(optimization_level = 3,
                  resilience_level = 0, 
                  transpilation = {'skip_transpilation':True}, 
                  execution = {'shots':shots})

# optimization_level not required in options, since anyway transpiled circuit...
# ...is given and transpilation in execution is skipped.

# Sampler run:
print('\nBeginning Sampler run:\n')
with Session(service = service, backend = 'ibmq_qasm_simulator') as session:
    sampler = Sampler(session = session, options = options)
    job = sampler.run(circuits = trot_qc_transpiled)  #parameters already bound to circuit
    print(job.job_id())
    job_monitor(job)
    #print(job.result())                

  
sampler_result = job.result()

prob_dict = sampler_result.quasi_dists[0].nearest_probability_distribution()

keys_list = list(prob_dict.keys())
for i in range(len(keys_list)):
    keys_list[i] = bin(keys_list[i])[2:].zfill(cutoff)

values_list = list(prob_dict.values())
for i in range(len(values_list)):
    values_list[i] = round(values_list[i], 4)

counts_probs = dict(zip(keys_list, values_list))

print(counts_probs)
##plot_histogram(counts_probs)

'''
print('Exact evolution:')
t = 0.2

exact_evolution_op = (t*H_lf).exp_i()

print(f'time = {t}')
#print('Exact evolution:')

exact_res = exact_evolution_op@init_state
#print('result: ', exact_res)
#print('result: ', exact_res.to_matrix())
#amplitudes_array = np.around(np.abs(exact_res.to_matrix()), 4)

probabilities = np.around((np.abs(exact_res.to_matrix()))**2, 4)
nonzeroindices = np.nonzero(probabilities)[0]
print(f'Indices of non-zero values: {nonzeroindices}')
exact_keys_list = [bin(ind)[2:].zfill(cutoff) for ind in nonzeroindices]
exact_prob_list = [probabilities[ind] for ind in nonzeroindices]
print(f'Non-zero probabilities: {exact_prob_list}')
exact_probs_dict = dict(zip(exact_keys_list, exact_prob_list))
print(f'Exact probabilities list: {exact_probs_dict}')

##plot_histogram(exact_probs_dict)
'''

#============================================================================
'''
Qiskit version:
    
{'qiskit-terra': '0.23.2', 
 'qiskit-aer': '0.11.2', 
 'qiskit-ignis': None, 
 'qiskit-ibmq-provider': '0.19.2', 
 'qiskit': '0.39.5', 
 'qiskit-nature': '0.4.5', 
 'qiskit-finance': '0.3.4', 
 'qiskit-optimization': '0.4.0', 
 'qiskit-machine-learning': '0.5.0'}

'''
#============================================================================

