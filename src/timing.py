import pstats
p = pstats.Stats('cputimes')
p.strip_dirs().sort_stats(-1).print_stats()


print(p.sort_stats('time').print_stats(10))


print(p.sort_stats('cumulative').print_stats(10))
