import unittest
import glob


def get_test_modules():
    test_files = glob.glob('*/*/unit_test/test_*.py')
    mod_names = []
    for test_file in test_files:
        mod_name = test_file.replace('.py', '').replace('/', '.')
        mod_names.append(mod_name)
    return mod_names


if __name__ == "__main__":
    test_modules = get_test_modules()
    suite = unittest.TestSuite()

    for t in test_modules:
        print t
        try:
            # If the module defines a suite() function, call it to get the suite.
            mod = __import__(t, globals(), locals(), ['suite'])
            suitefn = getattr(mod, 'suite')
            suite.addTest(suitefn())
        except (ImportError, AttributeError):
            # else, just load all the test cases from the module.
            suite.addTest(unittest.defaultTestLoader.loadTestsFromName(t))
    unittest.TextTestRunner().run(suite)
