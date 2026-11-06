import unittest

from data.make_cide_crud import ENTITIES, specs
from kelus.crud_tasks import reference, verify


class CrudTaskTests(unittest.TestCase):
    def test_varied_references_pass_ownership_validation_and_injection_checks(self):
        for s in specs('test'):
            with self.subTest(operation=s['operation'],table=s['table']):
                self.assertTrue(verify(reference(s),s))

    def test_checker_rejects_wrong_owner_and_missing_validation(self):
        rows=list(specs('test'))
        s=next(r for r in rows if r['operation']=='get')
        wrong=reference(s).replace(f"AND {s['owner']} = ?",'AND 1 = ?').replace(
            f"({s['item']}, {s['user']})).fetchone()",f"({s['item']}, 1)).fetchone()")
        with self.assertRaises(AssertionError):verify(wrong,s)
        s=next(r for r in rows if r['operation']=='update')
        wrong='\n'.join(line for line in reference(s).splitlines() if not line.startswith(('    if ', '        raise ')))
        with self.assertRaises(Exception):verify(wrong,s)
        s=next(r for r in rows if r['operation']=='create')
        with self.assertRaises(AssertionError):verify(reference(s).replace(f"    {s['db']}.commit()\n",''),s)

    def test_schema_groups_do_not_cross_splits(self):
        groups=[set(ENTITIES[s]) for s in ('train','validation','test')]
        self.assertFalse(groups[0]&groups[1] or groups[0]&groups[2] or groups[1]&groups[2])
        training={reference(s) for s in specs('train')}
        for split in ('validation','test'):
            self.assertFalse(training & {reference(s) for s in specs(split)})


if __name__=='__main__':unittest.main()
