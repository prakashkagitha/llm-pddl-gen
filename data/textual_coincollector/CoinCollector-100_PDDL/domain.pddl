(define (domain coin-collector)
  (:requirements :strips :typing)
  (:types
    room
    direction
    item
  )
  (:predicates
    (at ?room - room)
    (connected ?room1 - room ?room2 - room ?direction - direction)
    (closed-door ?room1 - room ?room2 - room ?direction - direction)
    (location ?item - item ?room - room)
    (taken ?item - item)
    (is-reverse ?direction - direction ?reverse - direction)
  )

  (:action move
    :parameters (?room1 - room ?room2 - room ?direction - direction)
    :precondition (and (at ?room1) (connected ?room1 ?room2 ?direction))
    :effect (and (not (at ?room1)) (at ?room2))
  )

  (:action open
    :parameters (?room1 - room ?room2 - room ?direction - direction ?reverse - direction)
    :precondition (and (at ?room1) (closed-door ?room1 ?room2 ?direction) (is-reverse ?direction ?reverse))
    :effect (and (not (closed-door ?room1 ?room2 ?direction)) (not (closed-door ?room2 ?room1 ?reverse)) (connected ?room1 ?room2 ?direction) (connected ?room2 ?room1 ?reverse))
  )
  
  (:action take
    :parameters (?item - item ?room - room)
    :precondition (and (at ?room) (location ?item ?room) (not (taken ?item)))
    :effect (and (taken ?item) (not (location ?item ?room)))
  )
)