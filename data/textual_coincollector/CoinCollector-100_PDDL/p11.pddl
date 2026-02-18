(define (problem coin_collector_numLocations7_numDistractorItems0_seed80)
  (:domain coin-collector)
  (:objects
    kitchen pantry corridor backyard living_room bathroom bedroom - room
    north south east west - direction
    coin - item
  )
  (:init
    (at kitchen)
    (closed-door kitchen pantry north)
    (connected kitchen corridor south)
    (closed-door kitchen backyard east)
    (connected kitchen living_room west)
    (closed-door pantry kitchen south)
    (connected corridor kitchen north)
    (closed-door corridor bathroom east)
    (closed-door corridor bedroom west)
    (closed-door backyard kitchen west)
    (connected living_room kitchen east)
    (closed-door living_room bedroom south)
    (closed-door bathroom corridor west)
    (closed-door bedroom corridor east)
    (closed-door bedroom living_room north)
    (location coin bathroom)
    (is-reverse north south)
    (is-reverse south north)
    (is-reverse east west)
    (is-reverse west east)
  )
  (:goal 
    (taken coin)
  )
)