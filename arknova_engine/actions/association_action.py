from __future__ import annotations

from typing import List

from arknova_engine.base_game import (
    AssociationTaskSelection,
    DONATION_COST_TRACK,
    GameState,
    MainAction,
)


def run_association_action(
    gs: GameState,
    tasks: List[AssociationTaskSelection],
    make_donation: bool = False,
    x_spent: int = 0,
) -> None:
    player_id = gs.current_player
    player = gs.current()
    base_strength = gs._slot_strength(player, MainAction.ASSOCIATION)
    gs._consume_x_tokens_for_action(player, x_spent)
    strength = base_strength + x_spent
    is_association_upgraded = player.action_upgraded[MainAction.ASSOCIATION]

    if not tasks:
        raise ValueError("Association action requires at least one association task.")
    if not is_association_upgraded and len(tasks) != 1:
        raise ValueError("Association side I requires exactly one association task.")

    task_types = [task.task for task in tasks]
    if len(task_types) != len(set(task_types)):
        raise ValueError("Association side II allows only different tasks in one action.")

    total_value = sum(gs._association_task_value(task.task) for task in tasks)
    if total_value > strength:
        raise ValueError(
            f"Association tasks total value {total_value} exceeds action strength {strength}."
        )

    took_from_display = False
    workers_spent = 0
    for task in tasks:
        required_workers = gs._association_workers_needed(player, task.task)
        if player.active_workers < required_workers:
            raise ValueError("Not enough active association workers.")
        task_took_from_display = gs._execute_association_task(
            player_id=player_id,
            player=player,
            task=task,
            is_association_upgraded=is_association_upgraded,
        )
        gs._spend_association_workers(player, task.task, required_workers)
        workers_spent += required_workers
        took_from_display = took_from_display or task_took_from_display

    if make_donation:
        if not is_association_upgraded:
            raise ValueError("Donation is only available with upgraded Association action.")
        donation_cost = gs._current_donation_cost()
        if player.money < donation_cost:
            raise ValueError("Insufficient money for donation.")
        if gs.donation_progress < len(DONATION_COST_TRACK) - 1:
            if player.player_tokens_in_supply <= 0:
                raise ValueError("No player tokens in supply for donation.")
            player.player_tokens_in_supply -= 1
        player.money -= donation_cost
        player.conservation += 1
        gs.donation_progress += 1

    if took_from_display:
        gs._replenish_display_end_of_turn()

    gs.action_log.append(
        "P{} ASSOCIATION tasks={} total_value={} workers={} strength={} upgraded={} donation={}".format(
            player_id,
            [task.task.value for task in tasks],
            total_value,
            workers_spent,
            strength,
            is_association_upgraded,
            make_donation,
        )
    )
    gs._complete_main_action(player_id, MainAction.ASSOCIATION)
