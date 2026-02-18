(define (problem coin_collector_numLocations7_numDistractorItems0_seed72)
  (:domain coin-collector)
  (:objects
    kitchen corridor backyard pantry bedroom bathroom living_room - room
    north south east west - direction
    coin - item
  )
  (:init
    (at kitchen)
    (connected kitchen corridor north)
    (closed-door kitchen backyard east)
    (closed-door kitchen pantry west)
    (connected corridor kitchen south)
    (closed-door corridor bedroom north)
    (closed-door corridor bathroom west)
    (closed-door backyard kitchen west)
    (closed-door pantry kitchen east)
    (closed-door bedroom corridor south)
    (closed-door bedroom living_room west)
    (closed-door bathroom corridor east)
    (closed-door bathroom living_room north)
    (closed-door living_room bedroom east)
    (closed-door living_room bathroom south)
    (location coin bedroom)
    (is-reverse north south)
    (is-reverse south north)
    (is-reverse east west)
    (is-reverse west east)
  )
  (:goal 
    (taken coin)
  )
)