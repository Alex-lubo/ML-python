# coding:utf-8
from math import sqrt
from numpy import np

critics = {'Lisa Rose': {'Lady in the Water': 2.5, 'Snakes on a Plane': 3.5,
                         'Just My Luck': 3.0, 'Superman Returns': 3.5, 'You, Me and Dupree': 2.5,
                         'The Night Listener': 3.0},
           'Gene Seymour': {'Lady in the Water': 3.0, 'Snakes on a Plane': 3.5,
                            'Just My Luck': 1.5, 'Superman Returns': 5.0, 'The Night Listener': 3.0,
                            'You, Me and Dupree': 3.5},
           'Michael Phillips': {'Lady in the Water': 2.5, 'Snakes on a Plane': 3.0,
                                'Superman Returns': 3.5, 'The Night Listener': 4.0},
           'Claudia Puig': {'Snakes on a Plane': 3.5, 'Just My Luck': 3.0,
                            'The Night Listener': 4.5, 'Superman Returns': 4.0,
                            'You, Me and Dupree': 2.5},
           'Mick LaSalle': {'Lady in the Water': 3.0, 'Snakes on a Plane': 4.0,
                            'Just My Luck': 2.0, 'Superman Returns': 3.0, 'The Night Listener': 3.0,
                            'You, Me and Dupree': 2.0},
           'Jack Matthews': {'Lady in the Water': 3.0, 'Snakes on a Plane': 4.0,
                             'The Night Listener': 3.0, 'Superman Returns': 5.0, 'You, Me and Dupree': 3.5},
           'Toby': {'Snakes on a Plane': 4.5, 'You, Me and Dupree': 1.0, 'Superman Returns': 4.0}}

def eu_distance(matrix, per1, per2):
  si = {}
  for item in matrix[per1]:
    if item in matrix[per2]:
      si[item] = 1
  if len(si) == 0:
    return 0
  
  sum_sqrt = sum([pow(matrix[per1][item] - matrix[per2][item], 2) for item in matrix[per1] if item in matrix[per2]])

  return 1/(1+sum_sqrt)

def pearson_relation(matrix, per1, per2):
  si = {}
  for item in matrix[per1]:
    if item in matrix[per2]: si[item] = 1
  
  n = len(si)
  if n == 0:return 0
  sum1 = sum([matrix[per1][it] for it in si])
  sum2 = sum([matrix[per2][it] for it in si])

  sum1sq = sum([pow(matrix[per1][it], 2) for it in si])
  sum2sq = sum([pow(matrix[per2][it], 2) for it in si])
  pSum = sum([matrix[per1][it]*matrix[per2][it] for it in si])

  num = pSum - (sum1*sum2/n)
  den = sqrt((sum1sq - pow(sum1,2)/n)*sqrt(sum2sq - pow(sum2, 2)/n))
  if den == 0: return 0
  return num/den

if __name__ == '__main__':
  print eu_distance(critics, 'Lisa Rose', 'Toby')