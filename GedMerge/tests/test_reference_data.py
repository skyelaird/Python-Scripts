"""Tests for multilingual reference data loader (noble titles, place names)."""

import unittest
from gedmerge.data.reference_loader import reference_data, ReferenceDataLoader


class TestReferenceDataLoader(unittest.TestCase):
    """Test the reference data loader singleton."""

    def test_singleton_pattern(self):
        """Test that ReferenceDataLoader uses singleton pattern."""
        loader1 = ReferenceDataLoader()
        loader2 = ReferenceDataLoader()
        self.assertIs(loader1, loader2)

    def test_data_loaded(self):
        """Test that reference data is loaded on initialization."""
        self.assertGreater(len(reference_data.titles), 0)
        self.assertGreater(len(reference_data.place_variants), 0)


class TestNobleTitleNormalization(unittest.TestCase):
    """Test multilingual noble title normalization."""

    def test_duke_variants(self):
        """Test that duke/duc/herzog normalize to same canonical form."""
        self.assertEqual(reference_data.normalize_suffix('duke'), 'duke')
        self.assertEqual(reference_data.normalize_suffix('duc'), 'duke')
        self.assertEqual(reference_data.normalize_suffix('herzog'), 'duke')
        self.assertEqual(reference_data.normalize_suffix('duca'), 'duke')
        self.assertEqual(reference_data.normalize_suffix('duque'), 'duke')

    def test_count_variants(self):
        """Test that count/comte/graf normalize to same canonical form."""
        self.assertEqual(reference_data.normalize_suffix('count'), 'count')
        self.assertEqual(reference_data.normalize_suffix('comte'), 'count')
        self.assertEqual(reference_data.normalize_suffix('graf'), 'count')
        self.assertEqual(reference_data.normalize_suffix('conte'), 'count')
        self.assertEqual(reference_data.normalize_suffix('conde'), 'count')

    def test_prince_variants(self):
        """Test that prince/prinz normalize to same canonical form."""
        self.assertEqual(reference_data.normalize_suffix('prince'), 'prince')
        self.assertEqual(reference_data.normalize_suffix('prinz'), 'prince')
        self.assertEqual(reference_data.normalize_suffix('fürst'), 'prince')
        self.assertEqual(reference_data.normalize_suffix('principe'), 'prince')

    def test_baron_variants(self):
        """Test that baron/freiherr normalize to same canonical form."""
        self.assertEqual(reference_data.normalize_suffix('baron'), 'baron')
        self.assertEqual(reference_data.normalize_suffix('freiherr'), 'baron')
        self.assertEqual(reference_data.normalize_suffix('barone'), 'baron')
        self.assertEqual(reference_data.normalize_suffix('barón'), 'baron')

    def test_case_insensitive(self):
        """Test that normalization is case-insensitive."""
        self.assertEqual(reference_data.normalize_suffix('DUKE'), 'duke')
        self.assertEqual(reference_data.normalize_suffix('Herzog'), 'duke')
        self.assertEqual(reference_data.normalize_suffix('DUC'), 'duke')

    def test_whitespace_handling(self):
        """Test that normalization handles whitespace."""
        self.assertEqual(reference_data.normalize_suffix('  duke  '), 'duke')
        self.assertEqual(reference_data.normalize_suffix('comte '), 'count')

    def test_professional_suffixes(self):
        """Test professional suffix normalization."""
        self.assertEqual(reference_data.normalize_suffix('dr'), 'doctor')
        self.assertEqual(reference_data.normalize_suffix('doctor'), 'doctor')
        self.assertEqual(reference_data.normalize_suffix('doktor'), 'doctor')
        self.assertEqual(reference_data.normalize_suffix('dottore'), 'doctor')

    def test_generational_suffixes(self):
        """Test generational suffix normalization."""
        self.assertEqual(reference_data.normalize_suffix('jr'), 'junior')
        self.assertEqual(reference_data.normalize_suffix('junior'), 'junior')
        self.assertEqual(reference_data.normalize_suffix('sr'), 'senior')
        self.assertEqual(reference_data.normalize_suffix('ii'), 'second')
        self.assertEqual(reference_data.normalize_suffix('iii'), 'third')

    def test_unknown_suffix(self):
        """Test that unknown suffixes return None."""
        self.assertIsNone(reference_data.normalize_suffix('xyz123'))
        self.assertIsNone(reference_data.normalize_suffix('unknown'))


class TestSuffixEquivalence(unittest.TestCase):
    """Test suffix equivalence checking."""

    def test_equivalent_noble_titles(self):
        """Test that equivalent titles in different languages are recognized."""
        self.assertTrue(reference_data.are_equivalent_suffixes('duke', 'duc'))
        self.assertTrue(reference_data.are_equivalent_suffixes('herzog', 'duca'))
        self.assertTrue(reference_data.are_equivalent_suffixes('count', 'graf'))
        self.assertTrue(reference_data.are_equivalent_suffixes('comte', 'conde'))

    def test_non_equivalent_titles(self):
        """Test that different titles are not equivalent."""
        self.assertFalse(reference_data.are_equivalent_suffixes('duke', 'count'))
        self.assertFalse(reference_data.are_equivalent_suffixes('prince', 'baron'))

    def test_case_insensitive_equivalence(self):
        """Test equivalence checking is case-insensitive."""
        self.assertTrue(reference_data.are_equivalent_suffixes('DUKE', 'duc'))
        self.assertTrue(reference_data.are_equivalent_suffixes('Herzog', 'DUCA'))


class TestPlaceNameNormalization(unittest.TestCase):
    """Test multilingual place name normalization."""

    def test_vienna_variants(self):
        """Test that Vienna/Wien/Bécs normalize to same canonical form."""
        self.assertEqual(reference_data.normalize_place('vienna'), 'vienna')
        self.assertEqual(reference_data.normalize_place('wien'), 'vienna')
        self.assertEqual(reference_data.normalize_place('bécs'), 'vienna')
        self.assertEqual(reference_data.normalize_place('vienne'), 'vienna')

    def test_munich_variants(self):
        """Test that Munich/München normalize to same canonical form."""
        self.assertEqual(reference_data.normalize_place('munich'), 'munich')
        self.assertEqual(reference_data.normalize_place('münchen'), 'munich')
        self.assertEqual(reference_data.normalize_place('monaco di baviera'), 'munich')

    def test_prague_variants(self):
        """Test that Prague/Prag/Praha normalize to same canonical form."""
        self.assertEqual(reference_data.normalize_place('prague'), 'prague')
        self.assertEqual(reference_data.normalize_place('prag'), 'prague')
        self.assertEqual(reference_data.normalize_place('praha'), 'prague')
        self.assertEqual(reference_data.normalize_place('praga'), 'prague')

    def test_rome_variants(self):
        """Test that Rome/Rom/Roma normalize to same canonical form."""
        self.assertEqual(reference_data.normalize_place('rome'), 'rome')
        self.assertEqual(reference_data.normalize_place('rom'), 'rome')
        self.assertEqual(reference_data.normalize_place('roma'), 'rome')

    def test_case_insensitive_places(self):
        """Test that place normalization is case-insensitive."""
        self.assertEqual(reference_data.normalize_place('VIENNA'), 'vienna')
        self.assertEqual(reference_data.normalize_place('Wien'), 'vienna')
        self.assertEqual(reference_data.normalize_place('BÉCS'), 'vienna')

    def test_regional_variants(self):
        """Test regional name normalization."""
        self.assertEqual(reference_data.normalize_place('bavaria'), 'bavaria')
        self.assertEqual(reference_data.normalize_place('bayern'), 'bavaria')
        self.assertEqual(reference_data.normalize_place('bavière'), 'bavaria')

    def test_country_variants(self):
        """Test country name normalization."""
        self.assertEqual(reference_data.normalize_place('germany'), 'germany')
        self.assertEqual(reference_data.normalize_place('deutschland'), 'germany')
        self.assertEqual(reference_data.normalize_place('allemagne'), 'germany')

    def test_unknown_place(self):
        """Test that unknown places return None."""
        self.assertIsNone(reference_data.normalize_place('xyz123city'))
        self.assertIsNone(reference_data.normalize_place('unknown place'))


class TestPlaceEquivalence(unittest.TestCase):
    """Test place name equivalence checking."""

    def test_equivalent_place_names(self):
        """Test that equivalent places in different languages are recognized."""
        self.assertTrue(reference_data.are_equivalent_places('vienna', 'wien'))
        self.assertTrue(reference_data.are_equivalent_places('münchen', 'munich'))
        self.assertTrue(reference_data.are_equivalent_places('prague', 'praha'))
        self.assertTrue(reference_data.are_equivalent_places('rome', 'roma'))

    def test_non_equivalent_places(self):
        """Test that different places are not equivalent."""
        self.assertFalse(reference_data.are_equivalent_places('vienna', 'munich'))
        self.assertFalse(reference_data.are_equivalent_places('rome', 'paris'))

    def test_case_insensitive_place_equivalence(self):
        """Test place equivalence checking is case-insensitive."""
        self.assertTrue(reference_data.are_equivalent_places('VIENNA', 'wien'))
        self.assertTrue(reference_data.are_equivalent_places('München', 'MUNICH'))


class TestTitleInfo(unittest.TestCase):
    """Test retrieving full title information."""

    def test_get_duke_info(self):
        """Test retrieving duke title information."""
        info = reference_data.get_title_info('duke')
        self.assertIsNotNone(info)
        self.assertEqual(info.canonical, 'duke')
        self.assertEqual(info.category, 'nobility')
        self.assertIsNotNone(info.rank)

    def test_get_title_rank(self):
        """Test that title ranks are properly ordered."""
        duke_rank = reference_data.get_suffix_rank('duke')
        count_rank = reference_data.get_suffix_rank('count')
        baron_rank = reference_data.get_suffix_rank('baron')

        # Lower rank number = higher rank
        self.assertLess(duke_rank, count_rank)
        self.assertLess(count_rank, baron_rank)

    def test_feminine_variants(self):
        """Test that feminine variants are included."""
        duchess_info = reference_data.get_title_info('duchess')
        self.assertIsNotNone(duchess_info)
        self.assertEqual(duchess_info.canonical, 'duke')  # Same canonical as duke


class TestPlaceInfo(unittest.TestCase):
    """Test retrieving full place information."""

    def test_get_vienna_info(self):
        """Test retrieving Vienna place information."""
        info = reference_data.get_place_info('vienna')
        self.assertIsNotNone(info)
        self.assertEqual(info.canonical, 'vienna')
        self.assertEqual(info.country, 'austria')
        self.assertEqual(info.place_type, 'city')

    def test_get_bavaria_info(self):
        """Test retrieving Bavaria region information."""
        info = reference_data.get_place_info('bavaria')
        self.assertIsNotNone(info)
        self.assertEqual(info.canonical, 'bavaria')
        self.assertEqual(info.country, 'germany')
        self.assertEqual(info.place_type, 'region')


class TestVariantRetrieval(unittest.TestCase):
    """Test retrieving all variants."""

    def test_get_all_suffix_variants(self):
        """Test retrieving all suffix variants."""
        variants = reference_data.get_all_suffix_variants()
        self.assertIn('duke', variants)
        self.assertIn('duc', variants)
        self.assertIn('herzog', variants)
        self.assertIn('count', variants)
        self.assertIn('graf', variants)
        self.assertIn('jr', variants)
        self.assertIn('sr', variants)

    def test_get_noble_title_variants(self):
        """Test retrieving only noble title variants."""
        variants = reference_data.get_noble_title_variants()
        self.assertIn('duke', variants)
        self.assertIn('count', variants)
        # Should include noble titles but also professional/generational
        # since get_noble_title_variants only returns titles with category='nobility' or 'royalty'


if __name__ == '__main__':
    unittest.main()
