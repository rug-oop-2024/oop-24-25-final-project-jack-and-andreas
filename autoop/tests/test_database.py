import unittest

from autoop.core.database import Database
from autoop.core.storage import LocalStorage
import random
import tempfile

"""
This class contains all the tests for the database class
It tests the different methods of the database class
"""


class TestDatabase(unittest.TestCase):
    """ Setup """
    def setUp(self):
        self.storage = LocalStorage(tempfile.mkdtemp())
        self.db = Database(self.storage)
    """ Check for succesful initailization """
    def test_init(self):
        self.assertIsInstance(self.db, Database)
    """ Test set by setting a key and checking if it is set """
    def test_set(self):
        id = str(random.randint(0, 100))
        entry = {"key": random.randint(0, 100)}
        self.db.set("collection", id, entry)
        self.assertEqual(self.db.get("collection", id)["key"], entry["key"])
    """ Test delete by setting and deleting a key """
    def test_delete(self):
        id = str(random.randint(0, 100))
        value = {"key": random.randint(0, 100)}
        self.db.set("collection", id, value)
        self.db.delete("collection", id)
        self.assertIsNone(self.db.get("collection", id))
        self.db.refresh()
        self.assertIsNone(self.db.get("collection", id))
    """
    Test if the database is persistent.
    Sets a key, creates a new database and check if the key is still there
    """
    def test_persistance(self):
        id = str(random.randint(0, 100))
        value = {"key": random.randint(0, 100)}
        self.db.set("collection", id, value)
        other_db = Database(self.storage)
        self.assertEqual(other_db.get("collection", id)["key"], value["key"])
        """
        Test refresh by setting a key, refreshing the database
        and checking if the key is still there
        """
    def test_refresh(self):
        key = str(random.randint(0, 100))
        value = {"key": random.randint(0, 100)}
        other_db = Database(self.storage)
        self.db.set("collection", key, value)
        other_db.refresh()
        self.assertEqual(other_db.get("collection", key)["key"], value["key"])
    """ Test list by setting a key, then check if it's in the list """
    def test_list(self):
        key = str(random.randint(0, 100))
        value = {"key": random.randint(0, 100)}
        self.db.set("collection", key, value)
        # collection should now contain the key
        self.assertIn((key, value), self.db.list("collection"))
