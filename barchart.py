import matplotlib.pyplot as plt
from matplotlib.ticker import FormatStrFormatter

methods = ['Random Split', 'Stratified Split', 'K-Fold', 'Stratified K-Fold']
accuracy = [99.91, 99.82, 99.80, 99.86]
patterns = ['//', '\\\\', 'xx', '..']

plt.figure(figsize=(7,3.5))

bars = plt.bar(
    methods,
    accuracy,
    width=0.9,
    color='black',
    edgecolor='white',
    linewidth=2.5
)

for bar, pattern in zip(bars, patterns):
    bar.set_hatch(pattern)

for bar in bars:
    height = bar.get_height()
    plt.text(
        bar.get_x() + bar.get_width()/2,
        height + 0.003,
        f'{height:.2f}%',
        ha='center',
        fontsize=10
    )

plt.ylabel('Accuracy (%)', fontsize=10, fontweight='bold')
plt.xlabel('Data Splitting Method', fontsize=10, fontweight='bold')

plt.ylim(99.75, 99.95)
plt.yticks([99.75, 99.80, 99.85, 99.90, 99.95])

ax = plt.gca()
ax.yaxis.set_major_formatter(FormatStrFormatter('%.2f'))

plt.grid(axis='y', linestyle='--', linewidth=0.5)

plt.tight_layout()
plt.savefig('barchart.png', dpi=300, bbox_inches='tight')
plt.show()