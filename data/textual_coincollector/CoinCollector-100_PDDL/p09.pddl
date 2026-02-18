(define (problem coin_collector_numLocations7_numDistractorItems0_seed98)
  (:domain coin-collector)
  (:objects
    kitchen living_room bathroom pantry backyard corridor bedroom - room
    north south east west - direction
    coin - item
  )
  (:init
    (at kitchen)
    (connected kitchen living_room north)
    (closed-door kitchen bathroom south)
    (closed-door kitchen pantry east)
    (closed-door kitchen backyard west)
    (connected living_room kitchen south)
    (closed-door bathroom kitchen north)
    (closed-door pantry kitchen west)
    (closed-door backyard kitchen east)
    (closed-door backyard corridor west)
    (closed-door corridor backyard east)
    (closed-door corridor bedroom south)
    (closed-door bedroom corridor north)
    (location coin living_room)
    (is-reverse north south)
    (is-reverse south north)
    (is-reverse east west)
    (is-reverse west east)
  )
  (:goal 
    (taken coin)
  )
)