import numpy as np
import matplotlib.pyplot as plt
import os

# These data comes from running RRT* adversarial and the original heuristic
rrt_heli = 0.016170672427761275
rrt = [0.9959168241965973,0.002959762354847421,0.0005184985147177964,0.00030245746691871453,0.00021604104779908182,7.561436672967863e-05]

heuristic_heli = 0.1716967073813929
heuristic = [0.8948170348851812,0.03784878902293029,0.022612160838516856,0.015897998308901706,0.011100971962929786,0.017714673207812538]

# fig = plt.figure()
# langs = ['0', '1', '2', '3', '4', '5']
# x = np.arange(len(langs))  # the label locations
# fig = plt.figure()
# ax = fig.add_axes([0,0,1,1])
# ax.bar(x + 0.00, rrt, color = 'b', width = 0.25)
# ax.bar(x + 0.25, heuristic, color = 'g', width = 0.25)
# plt.legend()
# plt.savefig("figures/rrt_heuristic_comparison.png")


fig, ax = plt.subplots(figsize=(10,5))
x = np.arange(len(rrt))
width = 0.4
plt.bar(x-0.2, rrt,
        width, color='tab:red', label='rrt')
plt.bar(x+0.2, heuristic,
        width, color='gold', label='heuristic')
plt.title('Percentage of Detections by Search Parties', fontsize=25)
plt.xlabel(None)
# plt.xticks(top5_alcohol.index, top5_alcohol['country'], fontsize=17)
plt.ylabel('Percentage of Timesteps Detected', fontsize=20)
plt.yticks(fontsize=17)
# sns.despine(bottom=True)
ax.grid(False)
ax.tick_params(bottom=False, left=True)
plt.legend(frameon=False, fontsize=15)
# plt.show()
plt.savefig("figures/rrt_heuristic_comparison.png")