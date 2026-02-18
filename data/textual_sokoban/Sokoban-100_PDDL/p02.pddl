(define (problem sokoban-002)
(:domain sokoban)
(:objects
	dir-down - direction
	dir-left - direction
	dir-right - direction
	dir-up - direction
	player-01 - thing
	stone-01 - thing
	pos-1-1 - location
	pos-1-2 - location
	pos-1-3 - location
	pos-1-4 - location
	pos-1-5 - location
	pos-2-1 - location
	pos-2-2 - location
	pos-2-3 - location
	pos-2-4 - location
	pos-2-5 - location
	pos-3-1 - location
	pos-3-2 - location
	pos-3-3 - location
	pos-3-4 - location
	pos-3-5 - location
	pos-4-1 - location
	pos-4-2 - location
	pos-4-3 - location
	pos-4-4 - location
	pos-4-5 - location
)
(:goal (and
	(at-goal stone-01)
))
(:init
	(at player-01 pos-1-5)
	(at stone-01 pos-2-2)
	(clear pos-1-1)
	(clear pos-1-2)
	(clear pos-1-3)
	(clear pos-1-4)
	(clear pos-2-1)
	(clear pos-2-3)
	(clear pos-2-4)
	(clear pos-2-5)
	(clear pos-3-1)
	(clear pos-3-2)
	(clear pos-3-3)
	(clear pos-3-4)
	(clear pos-3-5)
	(clear pos-4-1)
	(clear pos-4-2)
	(clear pos-4-3)
	(clear pos-4-4)
	(clear pos-4-5)
	(is-goal pos-4-2)
	(is-nongoal pos-1-1)
	(is-nongoal pos-1-2)
	(is-nongoal pos-1-3)
	(is-nongoal pos-1-4)
	(is-nongoal pos-1-5)
	(is-nongoal pos-2-1)
	(is-nongoal pos-2-2)
	(is-nongoal pos-2-3)
	(is-nongoal pos-2-4)
	(is-nongoal pos-2-5)
	(is-nongoal pos-3-1)
	(is-nongoal pos-3-2)
	(is-nongoal pos-3-3)
	(is-nongoal pos-3-4)
	(is-nongoal pos-3-5)
	(is-nongoal pos-4-1)
	(is-nongoal pos-4-3)
	(is-nongoal pos-4-4)
	(is-nongoal pos-4-5)
	(is-player player-01)
	(is-stone stone-01)
	(move dir-down)
	(move dir-left)
	(move dir-right)
	(move dir-up)
	(move-dir pos-1-1 pos-1-2 dir-right)
	(move-dir pos-1-1 pos-2-1 dir-down)
	(move-dir pos-1-2 pos-1-1 dir-left)
	(move-dir pos-1-2 pos-1-3 dir-right)
	(move-dir pos-1-2 pos-2-2 dir-down)
	(move-dir pos-1-3 pos-1-2 dir-left)
	(move-dir pos-1-3 pos-1-4 dir-right)
	(move-dir pos-1-3 pos-2-3 dir-down)
	(move-dir pos-1-4 pos-1-3 dir-left)
	(move-dir pos-1-4 pos-1-5 dir-right)
	(move-dir pos-1-4 pos-2-4 dir-down)
	(move-dir pos-1-5 pos-1-4 dir-left)
	(move-dir pos-1-5 pos-2-5 dir-down)
	(move-dir pos-2-1 pos-1-1 dir-up)
	(move-dir pos-2-1 pos-2-2 dir-right)
	(move-dir pos-2-1 pos-3-1 dir-down)
	(move-dir pos-2-2 pos-1-2 dir-up)
	(move-dir pos-2-2 pos-2-1 dir-left)
	(move-dir pos-2-2 pos-2-3 dir-right)
	(move-dir pos-2-2 pos-3-2 dir-down)
	(move-dir pos-2-3 pos-1-3 dir-up)
	(move-dir pos-2-3 pos-2-2 dir-left)
	(move-dir pos-2-3 pos-2-4 dir-right)
	(move-dir pos-2-3 pos-3-3 dir-down)
	(move-dir pos-2-4 pos-1-4 dir-up)
	(move-dir pos-2-4 pos-2-3 dir-left)
	(move-dir pos-2-4 pos-2-5 dir-right)
	(move-dir pos-2-4 pos-3-4 dir-down)
	(move-dir pos-2-5 pos-1-5 dir-up)
	(move-dir pos-2-5 pos-2-4 dir-left)
	(move-dir pos-2-5 pos-3-5 dir-down)
	(move-dir pos-3-1 pos-2-1 dir-up)
	(move-dir pos-3-1 pos-3-2 dir-right)
	(move-dir pos-3-1 pos-4-1 dir-down)
	(move-dir pos-3-2 pos-2-2 dir-up)
	(move-dir pos-3-2 pos-3-1 dir-left)
	(move-dir pos-3-2 pos-3-3 dir-right)
	(move-dir pos-3-2 pos-4-2 dir-down)
	(move-dir pos-3-3 pos-2-3 dir-up)
	(move-dir pos-3-3 pos-3-2 dir-left)
	(move-dir pos-3-3 pos-3-4 dir-right)
	(move-dir pos-3-3 pos-4-3 dir-down)
	(move-dir pos-3-4 pos-2-4 dir-up)
	(move-dir pos-3-4 pos-3-3 dir-left)
	(move-dir pos-3-4 pos-3-5 dir-right)
	(move-dir pos-3-4 pos-4-4 dir-down)
	(move-dir pos-3-5 pos-2-5 dir-up)
	(move-dir pos-3-5 pos-3-4 dir-left)
	(move-dir pos-3-5 pos-4-5 dir-down)
	(move-dir pos-4-1 pos-3-1 dir-up)
	(move-dir pos-4-1 pos-4-2 dir-right)
	(move-dir pos-4-2 pos-3-2 dir-up)
	(move-dir pos-4-2 pos-4-1 dir-left)
	(move-dir pos-4-2 pos-4-3 dir-right)
	(move-dir pos-4-3 pos-3-3 dir-up)
	(move-dir pos-4-3 pos-4-2 dir-left)
	(move-dir pos-4-3 pos-4-4 dir-right)
	(move-dir pos-4-4 pos-3-4 dir-up)
	(move-dir pos-4-4 pos-4-3 dir-left)
	(move-dir pos-4-4 pos-4-5 dir-right)
	(move-dir pos-4-5 pos-3-5 dir-up)
	(move-dir pos-4-5 pos-4-4 dir-left)
)
)
