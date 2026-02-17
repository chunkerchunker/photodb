"""Tests for MaintenanceUtilities.auto_associate_clusters."""

import pytest
from unittest.mock import MagicMock, call

from photodb.utils.maintenance import MaintenanceUtilities


@pytest.fixture
def mock_pool():
    """Create a mock ConnectionPool."""
    return MagicMock()


@pytest.fixture
def maintenance(mock_pool):
    """Create a MaintenanceUtilities instance with mocked pool and repo."""
    m = MaintenanceUtilities(mock_pool, collection_id=1)
    m.repo = MagicMock()
    # Default: no person constraints
    m.repo.get_cannot_links_for_clusters.return_value = {}
    return m


def _make_cluster(cid, person_id=None, verified=False, face_count=5):
    return {
        "id": cid,
        "person_id": person_id,
        "verified": verified,
        "face_count": face_count,
        "representative_detection_id": cid * 10,
    }


class TestAutoAssociateClusters:
    def test_no_similar_pairs_returns_zeros(self, maintenance):
        """Empty pairs → no action."""
        maintenance.repo.find_similar_cluster_pairs.return_value = []
        maintenance.repo.get_clusters_for_association.return_value = []

        result = maintenance.auto_associate_clusters(threshold=0.55)

        assert result["persons_created"] == 0
        assert result["persons_merged"] == 0
        assert result["clusters_linked"] == 0
        assert result["groups_found"] == 0
        maintenance.repo.create_person.assert_not_called()
        maintenance.repo.link_clusters_to_person.assert_not_called()

    def test_two_unlinked_clusters_creates_person(self, maintenance):
        """Two clusters with no person → creates person + links both."""
        maintenance.repo.find_similar_cluster_pairs.return_value = [
            {"cluster_id_1": 1, "cluster_id_2": 2, "cosine_distance": 0.3},
        ]
        maintenance.repo.get_clusters_for_association.return_value = [
            _make_cluster(1),
            _make_cluster(2),
        ]
        maintenance.repo.link_clusters_to_person.return_value = 2

        result = maintenance.auto_associate_clusters(threshold=0.55)

        assert result["persons_created"] == 1
        assert result["clusters_linked"] == 2
        assert result["groups_found"] == 1
        maintenance.repo.create_person.assert_called_once()
        maintenance.repo.link_clusters_to_person.assert_called_once()

    def test_one_linked_one_unlinked_uses_existing_person(self, maintenance):
        """Cluster with person_id anchors the group."""
        maintenance.repo.find_similar_cluster_pairs.return_value = [
            {"cluster_id_1": 1, "cluster_id_2": 2, "cosine_distance": 0.4},
        ]
        maintenance.repo.get_clusters_for_association.return_value = [
            _make_cluster(1, person_id=100),
            _make_cluster(2),
        ]
        maintenance.repo.link_clusters_to_person.return_value = 1

        result = maintenance.auto_associate_clusters(threshold=0.55)

        assert result["persons_created"] == 0
        assert result["clusters_linked"] == 1
        assert result["groups_found"] == 1
        maintenance.repo.create_person.assert_not_called()
        # Should link cluster 2 to person 100
        maintenance.repo.link_clusters_to_person.assert_called_once_with(
            [2], 100, 1
        )

    def test_different_persons_merges_into_verified(self, maintenance):
        """Verified cluster's person wins the merge."""
        maintenance.repo.find_similar_cluster_pairs.return_value = [
            {"cluster_id_1": 1, "cluster_id_2": 2, "cosine_distance": 0.35},
        ]
        maintenance.repo.get_clusters_for_association.return_value = [
            _make_cluster(1, person_id=100, verified=False, face_count=10),
            _make_cluster(2, person_id=200, verified=True, face_count=3),
        ]
        maintenance.repo.merge_persons.return_value = 1

        result = maintenance.auto_associate_clusters(threshold=0.55)

        assert result["persons_merged"] == 1
        assert result["groups_found"] == 1
        # Person 200 (verified) should be kept, person 100 removed
        maintenance.repo.merge_persons.assert_called_once_with(200, 100, 1)

    def test_complete_linkage_merges_full_clique(self, maintenance):
        """A~B, B~C, A~C (full clique) → one group {A, B, C}."""
        maintenance.repo.find_similar_cluster_pairs.return_value = [
            {"cluster_id_1": 1, "cluster_id_2": 2, "cosine_distance": 0.3},
            {"cluster_id_1": 1, "cluster_id_2": 3, "cosine_distance": 0.35},
            {"cluster_id_1": 2, "cluster_id_2": 3, "cosine_distance": 0.4},
        ]
        maintenance.repo.get_clusters_for_association.return_value = [
            _make_cluster(1),
            _make_cluster(2),
            _make_cluster(3),
        ]
        maintenance.repo.link_clusters_to_person.return_value = 3

        result = maintenance.auto_associate_clusters(threshold=0.55)

        # Full clique → one group of 3
        assert result["groups_found"] == 1
        assert result["persons_created"] == 1
        assert result["clusters_linked"] == 3

        args = maintenance.repo.link_clusters_to_person.call_args
        linked_ids = sorted(args[0][0])
        assert linked_ids == [1, 2, 3]

    def test_complete_linkage_prevents_chaining(self, maintenance):
        """A~B, B~C but NOT A~C → two groups, not one (no single-linkage chaining)."""
        maintenance.repo.find_similar_cluster_pairs.return_value = [
            {"cluster_id_1": 1, "cluster_id_2": 2, "cosine_distance": 0.3},
            {"cluster_id_1": 2, "cluster_id_2": 3, "cosine_distance": 0.4},
            # No (1,3) pair — they're too far apart
        ]
        maintenance.repo.get_clusters_for_association.return_value = [
            _make_cluster(1),
            _make_cluster(2),
            _make_cluster(3),
        ]
        maintenance.repo.link_clusters_to_person.return_value = 2

        result = maintenance.auto_associate_clusters(threshold=0.55)

        # Without A~C, complete-linkage keeps them separate:
        # {1,2} is a group, but 3 can't join because 1~3 is missing.
        # 3 could form {2,3} but 2 is already in {1,2} and merging
        # would require 1~3. So: one group {1,2}, cluster 3 left out.
        assert result["groups_found"] == 1
        assert result["persons_created"] == 1
        assert result["clusters_linked"] == 2

    def test_dry_run_makes_no_changes(self, maintenance):
        """Groups found but no mutations in dry_run mode."""
        maintenance.repo.find_similar_cluster_pairs.return_value = [
            {"cluster_id_1": 1, "cluster_id_2": 2, "cosine_distance": 0.3},
        ]
        maintenance.repo.get_clusters_for_association.return_value = [
            _make_cluster(1),
            _make_cluster(2),
        ]

        result = maintenance.auto_associate_clusters(threshold=0.55, dry_run=True)

        assert result["groups_found"] == 1
        assert result["persons_created"] == 0
        assert result["clusters_linked"] == 0
        maintenance.repo.create_person.assert_not_called()
        maintenance.repo.link_clusters_to_person.assert_not_called()
        maintenance.repo.merge_persons.assert_not_called()

    def test_already_linked_same_person_no_action(self, maintenance):
        """Both clusters share person_id → no new links needed."""
        maintenance.repo.find_similar_cluster_pairs.return_value = [
            {"cluster_id_1": 1, "cluster_id_2": 2, "cosine_distance": 0.25},
        ]
        maintenance.repo.get_clusters_for_association.return_value = [
            _make_cluster(1, person_id=100),
            _make_cluster(2, person_id=100),
        ]

        result = maintenance.auto_associate_clusters(threshold=0.55)

        assert result["groups_found"] == 1
        assert result["persons_created"] == 0
        assert result["persons_merged"] == 0
        assert result["clusters_linked"] == 0
        maintenance.repo.create_person.assert_not_called()
        maintenance.repo.link_clusters_to_person.assert_not_called()
        maintenance.repo.merge_persons.assert_not_called()

    def test_cannot_link_prevents_grouping(self, maintenance):
        """Cluster with cannot-link to candidate person is excluded from group."""
        maintenance.repo.find_similar_cluster_pairs.return_value = [
            {"cluster_id_1": 1, "cluster_id_2": 2, "cosine_distance": 0.3},
        ]
        maintenance.repo.get_clusters_for_association.return_value = [
            _make_cluster(1, person_id=100),
            _make_cluster(2),
        ]
        # Cluster 2 has a cannot-link to person 100
        maintenance.repo.get_cannot_links_for_clusters.return_value = {2: {100}}

        result = maintenance.auto_associate_clusters(threshold=0.55)

        # Group should be filtered down to just cluster 1 (< 2 members), so skipped
        assert result["clusters_linked"] == 0
        assert result["persons_created"] == 0
        maintenance.repo.link_clusters_to_person.assert_not_called()

