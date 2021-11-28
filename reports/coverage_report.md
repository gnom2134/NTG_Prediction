## Coverage

Command: 
```shell
coverage run -m pytest .\test.py
```

Output:
```shell
=========================================================================== test session starts ===========================================================================
platform win32 -- Python 3.8.9, pytest-6.2.5, py-1.11.0, pluggy-1.0.0
rootdir: D:\my_projects\hse\ml_se_1.0\test
collected 2 items                                                                                                                                                          

test.py ..                                                                                                                                                           [100%]

============================================================================ warnings summary =============================================================================
test.py: 3500 warnings
  D:\my_projects\hse\ml_se_1.0\src\utils.py:85: RuntimeWarning: divide by zero encountered in log
    feature_list += [np.log(neighs_dist[k - 1]) * neighs_y[k - 1]]

test.py: 3500 warnings
  D:\my_projects\hse\ml_se_1.0\src\utils.py:87: RuntimeWarning: divide by zero encountered in double_scalars
    feature_list += [neighs_y[k - 1] / neighs_dist[k - 1]]

-- Docs: https://docs.pytest.org/en/stable/warnings.html
=============================================================== 2 passed, 7000 warnings in 60.23s (0:01:00) ===============================================================
```

Coverage:
```shell
Name                                           Stmts   Miss  Cover   Missing
----------------------------------------------------------------------------
D:\my_projects\hse\ml_se_1.0\src\__init__.py       0      0   100%
D:\my_projects\hse\ml_se_1.0\src\models.py        43      4    91%   18, 20, 22, 26
D:\my_projects\hse\ml_se_1.0\src\utils.py        103     13    87%   22, 57-58, 108-109, 159-168
__init__.py                                        0      0   100%
test.py                                           47      0   100%
----------------------------------------------------------------------------
TOTAL                                            193     17    91%
```

Файл ```main.py``` тестировать нет смысла, так как там только исполняемый код, который используется для предсказания значений на настоящем тесте. Итого: coverage = 91%.