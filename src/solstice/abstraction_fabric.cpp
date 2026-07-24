#include "rlf/solstice/abstraction_fabric.hpp"

#include <algorithm>
#include <bit>
#include <cctype>
#include <cmath>
#include <functional>
#include <limits>
#include <optional>
#include <queue>
#include <set>
#include <stdexcept>
#include <unordered_map>
#include <utility>

namespace rlf::solstice {
namespace {

constexpr std::uint64_t fnv_offset = 14'695'981'039'346'656'037ULL;
constexpr std::uint64_t fnv_prime = 1'099'511'628'211ULL;

void hash_u64(std::uint64_t& hash, const std::uint64_t value) noexcept {
    for (unsigned int index = 0U; index < 8U; ++index) {
        hash ^= (value >> (index * 8U)) & 0xFFULL;
        hash *= fnv_prime;
    }
}

void hash_string(std::uint64_t& hash, const std::string_view value) noexcept {
    hash_u64(hash, static_cast<std::uint64_t>(value.size()));
    for (const char character : value) {
        hash ^= static_cast<unsigned char>(character);
        hash *= fnv_prime;
    }
}

[[nodiscard]] std::string normalize(const std::string_view value) {
    std::string result;
    result.reserve(value.size());
    bool pending_space = false;
    for (const char raw : value) {
        const unsigned char character = static_cast<unsigned char>(raw);
        if (std::isspace(character) != 0) {
            pending_space = !result.empty();
            continue;
        }
        if (pending_space) {
            result.push_back(' ');
            pending_space = false;
        }
        result.push_back(static_cast<char>(std::tolower(character)));
    }
    while (!result.empty() && result.back() == ' ') {
        result.pop_back();
    }
    return result;
}

[[nodiscard]] bool is_variable(const std::string_view value) noexcept {
    return !value.empty() && value.front() == '?';
}

[[nodiscard]] std::string fact_key(
    const std::string_view subject,
    const std::string_view relation,
    const std::string_view object
) {
    std::string key;
    key.reserve(subject.size() + relation.size() + object.size() + 2U);
    key.append(subject);
    key.push_back('\x1f');
    key.append(relation);
    key.push_back('\x1f');
    key.append(object);
    return key;
}

using Bindings = std::map<std::string, std::string>;

[[nodiscard]] bool bind_term(
    const std::string_view pattern,
    const std::string_view value,
    Bindings& bindings
) {
    if (!is_variable(pattern)) {
        return pattern == value;
    }
    const auto iterator = bindings.find(std::string(pattern));
    if (iterator == bindings.end()) {
        bindings.emplace(pattern, value);
        return true;
    }
    return iterator->second == value;
}

[[nodiscard]] bool unify(
    const RelationalPattern& pattern,
    const ReasoningFact& fact,
    Bindings& bindings
) {
    Bindings candidate = bindings;
    if (!bind_term(pattern.subject, fact.subject, candidate) ||
        !bind_term(pattern.relation, fact.relation, candidate) ||
        !bind_term(pattern.object, fact.object, candidate)) {
        return false;
    }
    bindings = std::move(candidate);
    return true;
}

[[nodiscard]] std::string instantiate_term(
    const std::string_view term,
    const Bindings& bindings
) {
    if (!is_variable(term)) {
        return std::string(term);
    }
    const auto iterator = bindings.find(std::string(term));
    return iterator == bindings.end() ? std::string(term) : iterator->second;
}

[[nodiscard]] std::string statement(const ReasoningFact& fact) {
    return fact.subject + " --" + fact.relation + "--> " + fact.object;
}

void validate_probability(const double value, const char* const name) {
    if (!std::isfinite(value) || value < 0.0 || value > 1.0) {
        throw std::invalid_argument(std::string(name) + " must be in [0,1]");
    }
}

}  // namespace

AbstractionFabric::AbstractionFabric(AbstractionConfig config)
    : config_(std::move(config)) {
    if (config_.maximum_facts == 0U || config_.maximum_rules == 0U ||
        config_.maximum_inference_depth == 0U ||
        config_.maximum_derivations_per_query == 0U) {
        throw std::invalid_argument("invalid abstraction fabric capacity");
    }
    validate_probability(config_.minimum_confidence, "minimum confidence");
    validate_probability(
        config_.inferred_confidence_decay,
        "inferred confidence decay"
    );
}

std::string AbstractionFabric::canonical_relation(
    const std::string_view relation
) const {
    std::string current = normalize(relation);
    std::set<std::string> visited;
    while (true) {
        if (!visited.insert(current).second) {
            break;
        }
        const auto iterator = relation_aliases_.find(current);
        if (iterator == relation_aliases_.end() || iterator->second == current) {
            break;
        }
        current = iterator->second;
    }
    return current;
}

std::uint64_t AbstractionFabric::learn_fact(
    const std::string_view subject,
    const std::string_view relation,
    const std::string_view object,
    const double confidence,
    const std::string_view provenance
) {
    validate_probability(confidence, "fact confidence");
    const std::string normalized_subject = normalize(subject);
    const std::string normalized_relation = canonical_relation(relation);
    const std::string normalized_object = normalize(object);
    if (normalized_subject.empty() || normalized_relation.empty() ||
        normalized_object.empty()) {
        throw std::invalid_argument("fact terms must be non-empty");
    }
    const std::string key = fact_key(
        normalized_subject,
        normalized_relation,
        normalized_object
    );
    const auto existing = fact_index_.find(key);
    if (existing != fact_index_.end()) {
        ReasoningFact& fact = facts_[existing->second];
        fact.confidence = 1.0 -
            ((1.0 - fact.confidence) * (1.0 - confidence));
        ++fact.support;
        if (!provenance.empty()) {
            fact.provenance = std::string(provenance);
        }
        return fact.id;
    }
    if (facts_.size() >= config_.maximum_facts) {
        throw std::runtime_error("abstraction fact capacity exceeded");
    }
    ReasoningFact fact;
    fact.id = next_fact_id_++;
    fact.subject = normalized_subject;
    fact.relation = normalized_relation;
    fact.object = normalized_object;
    fact.confidence = confidence;
    fact.provenance = std::string(provenance);
    const std::uint64_t id = fact.id;
    fact_index_.emplace(key, facts_.size());
    facts_.push_back(std::move(fact));
    return id;
}

std::uint64_t AbstractionFabric::learn_rule(
    const std::string_view name,
    const std::span<const RelationalPattern> premises,
    const RelationalPattern& conclusion,
    const double confidence
) {
    validate_probability(confidence, "rule confidence");
    if (premises.empty()) {
        throw std::invalid_argument("reasoning rule requires at least one premise");
    }
    if (rules_.size() >= config_.maximum_rules) {
        throw std::runtime_error("abstraction rule capacity exceeded");
    }
    ReasoningRule rule;
    rule.id = next_rule_id_++;
    rule.name = normalize(name);
    if (rule.name.empty()) {
        rule.name = "rule_" + std::to_string(rule.id);
    }
    rule.premises.reserve(premises.size());
    for (const RelationalPattern& raw : premises) {
        RelationalPattern pattern{
            normalize(raw.subject),
            is_variable(raw.relation)
                ? normalize(raw.relation)
                : canonical_relation(raw.relation),
            normalize(raw.object),
        };
        if (pattern.subject.empty() || pattern.relation.empty() ||
            pattern.object.empty()) {
            throw std::invalid_argument("rule premise terms must be non-empty");
        }
        rule.premises.push_back(std::move(pattern));
    }
    rule.conclusion = {
        normalize(conclusion.subject),
        is_variable(conclusion.relation)
            ? normalize(conclusion.relation)
            : canonical_relation(conclusion.relation),
        normalize(conclusion.object),
    };
    if (rule.conclusion.subject.empty() || rule.conclusion.relation.empty() ||
        rule.conclusion.object.empty()) {
        throw std::invalid_argument("rule conclusion terms must be non-empty");
    }
    rule.confidence = confidence;
    const std::uint64_t id = rule.id;
    rules_.push_back(std::move(rule));
    return id;
}

void AbstractionFabric::learn_relation_equivalence(
    const std::string_view left,
    const std::string_view right
) {
    const std::string normalized_left = canonical_relation(left);
    const std::string normalized_right = canonical_relation(right);
    if (normalized_left.empty() || normalized_right.empty()) {
        throw std::invalid_argument("relation aliases must be non-empty");
    }
    const std::string canonical = std::min(normalized_left, normalized_right);
    relation_aliases_[normalized_left] = canonical;
    relation_aliases_[normalized_right] = canonical;
    relation_aliases_[canonical] = canonical;
    for (ReasoningFact& fact : facts_) {
        fact.relation = canonical_relation(fact.relation);
    }
    for (ReasoningRule& rule : rules_) {
        for (RelationalPattern& premise : rule.premises) {
            if (!is_variable(premise.relation)) {
                premise.relation = canonical_relation(premise.relation);
            }
        }
        if (!is_variable(rule.conclusion.relation)) {
            rule.conclusion.relation = canonical_relation(rule.conclusion.relation);
        }
    }
    rebuild_indices();
}

std::uint64_t AbstractionFabric::transfer_rule(
    const std::uint64_t source_rule_id,
    const std::string_view new_name,
    const std::map<std::string, std::string>& relation_mapping
) {
    const auto iterator = std::find_if(
        rules_.begin(), rules_.end(),
        [source_rule_id](const ReasoningRule& rule) {
            return rule.id == source_rule_id;
        }
    );
    if (iterator == rules_.end()) {
        throw std::invalid_argument("source rule does not exist");
    }
    ReasoningRule transferred = *iterator;
    transferred.id = next_rule_id_++;
    transferred.name = normalize(new_name);
    transferred.support = 1U;
    const auto replace_relation = [&relation_mapping](std::string& relation) {
        const auto mapping = relation_mapping.find(relation);
        if (mapping != relation_mapping.end()) {
            relation = normalize(mapping->second);
        }
    };
    for (RelationalPattern& premise : transferred.premises) {
        replace_relation(premise.relation);
    }
    replace_relation(transferred.conclusion.relation);
    if (rules_.size() >= config_.maximum_rules) {
        throw std::runtime_error("abstraction rule capacity exceeded");
    }
    const std::uint64_t id = transferred.id;
    rules_.push_back(std::move(transferred));
    return id;
}

SchemaInductionResult AbstractionFabric::induce_chain_rule(
    const std::string_view name,
    const std::string_view demonstration_subject,
    const std::string_view conclusion_relation,
    const std::string_view demonstration_object,
    const std::size_t maximum_hops,
    const double confidence
) {
    validate_probability(confidence, "schema confidence");
    if (maximum_hops == 0U) {
        throw std::invalid_argument("schema induction requires at least one hop");
    }
    const std::string start = normalize(demonstration_subject);
    const std::string goal = normalize(demonstration_object);
    const std::string target_relation = canonical_relation(conclusion_relation);
    if (start.empty() || goal.empty() || target_relation.empty()) {
        throw std::invalid_argument("schema induction terms must be non-empty");
    }

    std::unordered_map<std::string, std::vector<const ReasoningFact*>> adjacency;
    adjacency.reserve(facts_.size());
    for (const ReasoningFact& fact : facts_) {
        if (fact.relation == target_relation) {
            continue;
        }
        adjacency[fact.subject].push_back(&fact);
    }
    for (auto& [subject, edges] : adjacency) {
        static_cast<void>(subject);
        std::sort(
            edges.begin(), edges.end(),
            [](const ReasoningFact* left, const ReasoningFact* right) {
                if (left->relation != right->relation) {
                    return left->relation < right->relation;
                }
                if (left->object != right->object) {
                    return left->object < right->object;
                }
                return left->id < right->id;
            }
        );
    }

    struct SearchNode final {
        std::string entity;
        std::vector<std::string> relations;
    };
    std::queue<SearchNode> frontier;
    frontier.push({start, {}});
    std::unordered_map<std::string, std::size_t> shallowest_depth;
    shallowest_depth.emplace(start, 0U);
    std::uint64_t edges_examined = 0U;
    std::vector<std::string> discovered_path;

    while (!frontier.empty()) {
        SearchNode node = std::move(frontier.front());
        frontier.pop();
        if (node.entity == goal && !node.relations.empty()) {
            discovered_path = std::move(node.relations);
            break;
        }
        if (node.relations.size() >= maximum_hops) {
            continue;
        }
        const auto edges = adjacency.find(node.entity);
        if (edges == adjacency.end()) {
            continue;
        }
        for (const ReasoningFact* edge : edges->second) {
            ++edges_examined;
            const std::size_t next_depth = node.relations.size() + 1U;
            const auto visited = shallowest_depth.find(edge->object);
            if (visited != shallowest_depth.end() &&
                visited->second < next_depth) {
                continue;
            }
            shallowest_depth[edge->object] = next_depth;
            SearchNode next;
            next.entity = edge->object;
            next.relations = node.relations;
            next.relations.push_back(edge->relation);
            frontier.push(std::move(next));
        }
    }
    if (discovered_path.empty()) {
        throw std::runtime_error(
            "no reusable relational path explains the demonstration"
        );
    }

    std::vector<RelationalPattern> premises;
    premises.reserve(discovered_path.size());
    for (std::size_t hop = 0U; hop < discovered_path.size(); ++hop) {
        premises.push_back({
            "?x" + std::to_string(hop),
            discovered_path[hop],
            "?x" + std::to_string(hop + 1U),
        });
    }
    const RelationalPattern conclusion{
        "?x0",
        target_relation,
        "?x" + std::to_string(discovered_path.size()),
    };
    const std::uint64_t rule_id = learn_rule(
        name, premises, conclusion, confidence
    );
    return {
        rule_id,
        discovered_path.size(),
        edges_examined,
        std::move(discovered_path),
    };
}

ReasoningQueryResult AbstractionFabric::infer_with_stats(
    const RelationalPattern& raw_query,
    const std::size_t maximum_answers
) const {
    ReasoningQueryResult output;
    if (maximum_answers == 0U) {
        return output;
    }
    const RelationalPattern query{
        normalize(raw_query.subject),
        is_variable(raw_query.relation)
            ? normalize(raw_query.relation)
            : canonical_relation(raw_query.relation),
        normalize(raw_query.object),
    };
    std::vector<ReasoningFact> working = facts_;
    std::map<std::string, std::vector<ReasoningProofStep>> proofs;
    proofs.clear();
    for (const ReasoningFact& fact : working) {
        proofs.emplace(
            fact_key(fact.subject, fact.relation, fact.object),
            std::vector<ReasoningProofStep>{
                {fact.id, 0U, statement(fact)}
            }
        );
    }

    const auto pair_key = [](const std::string_view first,
                             const std::string_view second) {
        std::string key;
        key.reserve(first.size() + second.size() + 1U);
        key.append(first);
        key.push_back('\x1f');
        key.append(second);
        return key;
    };
    struct Indices final {
        std::unordered_map<std::string, std::vector<std::size_t>> relation;
        std::unordered_map<std::string, std::vector<std::size_t>> subject;
        std::unordered_map<std::string, std::vector<std::size_t>> object;
        std::unordered_map<std::string, std::vector<std::size_t>> subject_relation;
        std::unordered_map<std::string, std::vector<std::size_t>> object_relation;
    } indices;
    const auto index_fact = [&indices, &pair_key](
        const ReasoningFact& fact,
        const std::size_t index
    ) {
        indices.relation[fact.relation].push_back(index);
        indices.subject[fact.subject].push_back(index);
        indices.object[fact.object].push_back(index);
        indices.subject_relation[pair_key(fact.subject, fact.relation)]
            .push_back(index);
        indices.object_relation[pair_key(fact.object, fact.relation)]
            .push_back(index);
    };
    indices.relation.reserve(working.size());
    indices.subject.reserve(working.size());
    indices.object.reserve(working.size());
    indices.subject_relation.reserve(working.size());
    indices.object_relation.reserve(working.size());
    for (std::size_t index = 0U; index < working.size(); ++index) {
        index_fact(working[index], index);
    }
    std::unordered_map<std::string, std::size_t> working_fact_index;
    working_fact_index.reserve(working.size() * 2U + 1U);
    for (std::size_t index = 0U; index < working.size(); ++index) {
        const ReasoningFact& fact = working[index];
        working_fact_index.emplace(
            fact_key(fact.subject, fact.relation, fact.object), index
        );
    }

    const auto bound_term = [](const std::string& term,
                               const Bindings& bindings)
        -> std::optional<std::string_view> {
        if (!is_variable(term)) {
            return std::string_view(term);
        }
        const auto iterator = bindings.find(term);
        if (iterator == bindings.end()) {
            return std::nullopt;
        }
        return std::string_view(iterator->second);
    };

    const auto candidates_for = [
        &indices, &pair_key, &bound_term, &output
    ](const RelationalPattern& premise,
      const Bindings& bindings) -> const std::vector<std::size_t>* {
        const auto subject = bound_term(premise.subject, bindings);
        const auto relation = bound_term(premise.relation, bindings);
        const auto object = bound_term(premise.object, bindings);
        ++output.stats.index_lookups;
        if (subject.has_value() && relation.has_value()) {
            const auto iterator = indices.subject_relation.find(
                pair_key(*subject, *relation)
            );
            return iterator == indices.subject_relation.end()
                ? nullptr : &iterator->second;
        }
        if (object.has_value() && relation.has_value()) {
            const auto iterator = indices.object_relation.find(
                pair_key(*object, *relation)
            );
            return iterator == indices.object_relation.end()
                ? nullptr : &iterator->second;
        }
        if (relation.has_value()) {
            const auto iterator = indices.relation.find(std::string(*relation));
            return iterator == indices.relation.end()
                ? nullptr : &iterator->second;
        }
        if (subject.has_value()) {
            const auto iterator = indices.subject.find(std::string(*subject));
            return iterator == indices.subject.end()
                ? nullptr : &iterator->second;
        }
        if (object.has_value()) {
            const auto iterator = indices.object.find(std::string(*object));
            return iterator == indices.object.end()
                ? nullptr : &iterator->second;
        }
        return nullptr;
    };

    std::size_t derivations = 0U;
    for (std::size_t depth = 0U;
         depth < config_.maximum_inference_depth &&
         derivations < config_.maximum_derivations_per_query;
         ++depth) {
        bool changed = false;
        const std::size_t fact_count_at_start = working.size();
        for (const ReasoningRule& rule : rules_) {
            struct Match final {
                Bindings bindings;
                double confidence{1.0};
                std::vector<ReasoningProofStep> proof;
            };
            std::vector<Match> matches{{}};
            for (const RelationalPattern& premise : rule.premises) {
                std::vector<Match> next_matches;
                for (const Match& match : matches) {
                    output.stats.naive_candidate_upper_bound +=
                        static_cast<std::uint64_t>(fact_count_at_start);
                    const std::vector<std::size_t>* candidates =
                        candidates_for(premise, match.bindings);
                    if (candidates == nullptr) {
                        continue;
                    }
                    for (const std::size_t fact_index : *candidates) {
                        if (fact_index >= fact_count_at_start) {
                            continue;
                        }
                        ++output.stats.candidate_facts_examined;
                        ++output.stats.unification_attempts;
                        const ReasoningFact& fact = working[fact_index];
                        Match candidate = match;
                        if (!unify(premise, fact, candidate.bindings)) {
                            continue;
                        }
                        candidate.confidence = std::min(
                            candidate.confidence, fact.confidence
                        );
                        const auto proof_iterator = proofs.find(
                            fact_key(fact.subject, fact.relation, fact.object)
                        );
                        if (proof_iterator != proofs.end()) {
                            candidate.proof.insert(
                                candidate.proof.end(),
                                proof_iterator->second.begin(),
                                proof_iterator->second.end()
                            );
                        }
                        next_matches.push_back(std::move(candidate));
                        if (next_matches.size() >=
                            config_.maximum_derivations_per_query) {
                            break;
                        }
                    }
                    if (next_matches.size() >=
                        config_.maximum_derivations_per_query) {
                        break;
                    }
                }
                matches = std::move(next_matches);
                if (matches.empty()) {
                    break;
                }
            }
            for (const Match& match : matches) {
                ReasoningFact derived;
                derived.subject = instantiate_term(
                    rule.conclusion.subject, match.bindings
                );
                derived.relation = instantiate_term(
                    rule.conclusion.relation, match.bindings
                );
                derived.object = instantiate_term(
                    rule.conclusion.object, match.bindings
                );
                if (is_variable(derived.subject) ||
                    is_variable(derived.relation) ||
                    is_variable(derived.object)) {
                    continue;
                }
                derived.confidence = std::clamp(
                    match.confidence * rule.confidence *
                        std::pow(
                            config_.inferred_confidence_decay,
                            static_cast<double>(depth + 1U)
                        ),
                    0.0,
                    1.0
                );
                if (derived.confidence < config_.minimum_confidence) {
                    continue;
                }
                derived.inferred = true;
                derived.provenance = rule.name;
                const std::string key = fact_key(
                    derived.subject, derived.relation, derived.object
                );
                if (working_fact_index.contains(key)) {
                    continue;
                }
                derived.id = static_cast<std::uint64_t>(working.size()) +
                    next_fact_id_;
                std::vector<ReasoningProofStep> proof = match.proof;
                proof.push_back({derived.id, rule.id, statement(derived)});
                proofs.emplace(key, std::move(proof));
                working.push_back(std::move(derived));
                const std::size_t new_index = working.size() - 1U;
                working_fact_index.emplace(key, new_index);
                index_fact(working.back(), new_index);
                ++derivations;
                ++output.stats.derivations;
                changed = true;
                if (derivations >= config_.maximum_derivations_per_query) {
                    break;
                }
            }
            if (derivations >= config_.maximum_derivations_per_query) {
                break;
            }
        }
        if (!changed) {
            break;
        }
    }

    struct Candidate final {
        ReasoningAnswer answer;
        std::string key;
    };
    std::vector<Candidate> candidates;
    for (const ReasoningFact& fact : working) {
        Bindings bindings;
        ++output.stats.unification_attempts;
        if (!unify(query, fact, bindings)) {
            continue;
        }
        std::string value;
        if (is_variable(query.object)) {
            value = instantiate_term(query.object, bindings);
        } else if (is_variable(query.subject)) {
            value = instantiate_term(query.subject, bindings);
        } else if (is_variable(query.relation)) {
            value = instantiate_term(query.relation, bindings);
        } else {
            value = fact.object;
        }
        const std::string key = fact_key(
            fact.subject, fact.relation, fact.object
        );
        const auto proof_iterator = proofs.find(key);
        candidates.push_back({
            {
                value,
                fact.confidence,
                proof_iterator == proofs.end()
                    ? std::vector<ReasoningProofStep>{}
                    : proof_iterator->second,
                fact.subject,
                fact.relation,
                fact.object,
            },
            key,
        });
    }
    std::sort(
        candidates.begin(), candidates.end(),
        [](const Candidate& left, const Candidate& right) {
            if (left.answer.confidence != right.answer.confidence) {
                return left.answer.confidence > right.answer.confidence;
            }
            if (left.answer.value != right.answer.value) {
                return left.answer.value < right.answer.value;
            }
            return left.key < right.key;
        }
    );
    std::set<std::string> seen;
    const std::size_t query_variables =
        static_cast<std::size_t>(is_variable(query.subject)) +
        static_cast<std::size_t>(is_variable(query.relation)) +
        static_cast<std::size_t>(is_variable(query.object));
    for (Candidate& candidate : candidates) {
        const std::string unique_key = query_variables <= 1U
            ? candidate.answer.value
            : candidate.answer.matched_subject + "\x1f" +
                candidate.answer.matched_relation + "\x1f" +
                candidate.answer.matched_object;
        if (!seen.insert(unique_key).second) {
            continue;
        }
        output.answers.push_back(std::move(candidate.answer));
        if (output.answers.size() >= maximum_answers) {
            break;
        }
    }
    return output;
}

std::vector<ReasoningAnswer> AbstractionFabric::infer(
    const RelationalPattern& query,
    const std::size_t maximum_answers
) const {
    return infer_with_stats(query, maximum_answers).answers;
}

std::vector<ReasoningAnswer> AbstractionFabric::answer(
    const std::string_view subject,
    const std::string_view relation,
    const std::size_t maximum_answers
) const {
    return infer(
        RelationalPattern{
            std::string(subject), std::string(relation), "?answer"
        },
        maximum_answers
    );
}

std::span<const ReasoningFact> AbstractionFabric::facts() const noexcept {
    return facts_;
}

std::span<const ReasoningRule> AbstractionFabric::rules() const noexcept {
    return rules_;
}

const AbstractionConfig& AbstractionFabric::config() const noexcept {
    return config_;
}

std::uint64_t AbstractionFabric::deterministic_hash() const noexcept {
    std::uint64_t hash = fnv_offset;
    hash_u64(hash, next_fact_id_);
    hash_u64(hash, next_rule_id_);
    for (const ReasoningFact& fact : facts_) {
        hash_u64(hash, fact.id);
        hash_string(hash, fact.subject);
        hash_string(hash, fact.relation);
        hash_string(hash, fact.object);
        hash_u64(hash, std::bit_cast<std::uint64_t>(fact.confidence));
        hash_u64(hash, fact.support);
    }
    for (const ReasoningRule& rule : rules_) {
        hash_u64(hash, rule.id);
        hash_string(hash, rule.name);
        for (const RelationalPattern& premise : rule.premises) {
            hash_string(hash, premise.subject);
            hash_string(hash, premise.relation);
            hash_string(hash, premise.object);
        }
        hash_string(hash, rule.conclusion.subject);
        hash_string(hash, rule.conclusion.relation);
        hash_string(hash, rule.conclusion.object);
        hash_u64(hash, std::bit_cast<std::uint64_t>(rule.confidence));
    }
    return hash;
}

AbstractionSnapshot AbstractionFabric::snapshot() const {
    std::vector<std::pair<std::string, std::string>> aliases;
    aliases.reserve(relation_aliases_.size());
    for (const auto& [left, right] : relation_aliases_) {
        aliases.emplace_back(left, right);
    }
    return {
        config_, next_fact_id_, next_rule_id_, facts_, rules_,
        std::move(aliases),
    };
}

AbstractionFabric AbstractionFabric::from_snapshot(
    AbstractionSnapshot snapshot
) {
    AbstractionFabric fabric(snapshot.config);
    fabric.next_fact_id_ = snapshot.next_fact_id;
    fabric.next_rule_id_ = snapshot.next_rule_id;
    fabric.facts_ = std::move(snapshot.facts);
    fabric.rules_ = std::move(snapshot.rules);
    for (auto& [left, right] : snapshot.relation_aliases) {
        fabric.relation_aliases_.emplace(std::move(left), std::move(right));
    }
    fabric.rebuild_indices();
    return fabric;
}

void AbstractionFabric::rebuild_indices() {
    fact_index_.clear();
    for (std::size_t index = 0U; index < facts_.size(); ++index) {
        const ReasoningFact& fact = facts_[index];
        fact_index_[fact_key(fact.subject, fact.relation, fact.object)] = index;
    }
}

}  // namespace rlf::solstice
